from __future__ import annotations
import json, os, logging
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

from shared.models import QueryPlan, ThreadSelection
from shared.llm import call_llm_json, call_llm, schema_str
from shared.prompts import (
    QUERY_TO_FILTER_SYSTEM, QUERY_TO_FILTER_USER,
    QUERY_REFINE_SYSTEM, QUERY_REFINE_USER,
    THREAD_SELECT_SYSTEM, THREAD_SELECT_USER,
    AGENT_ANSWER_SYSTEM, AGENT_ANSWER_USER,
)
from chat_service.agent_tools import (
    search_threads,
    semantic_search_threads,
    bm25_search_threads,
    hybrid_search_threads,
    get_thread_detail,
    semantic_search_details,
    bm25_search_details,
    hybrid_search_details,
    get_daily_summaries,
    bm25_search_summaries,
    hybrid_search_summaries,
)

logger = logging.getLogger("email_classifier.langchain_agent")
@tool("build_plan")
def tool_build_plan(question: str) -> Dict[str, Any]:
    """Build a Weaviate filter plan for the question."""
    logger.info("tool_build_plan: question=%s", question)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    schema = schema_str(QueryPlan)
    user = QUERY_TO_FILTER_USER.format(question=question, schema=schema)
    raw = call_llm_json(
        model=model,
        system=QUERY_TO_FILTER_SYSTEM,
        user=user,
        schema=schema,
        max_tokens=400,
        temperature=0.0,
    )
    return QueryPlan.model_validate(raw).model_dump()


@tool("refine")
def tool_refine(question: str, plan: Optional[Any] = None) -> Dict[str, Any]:
    """Refine an existing Weaviate filter plan to increase recall."""
    logger.info("tool_refine: question=%s has_plan=%s", question, bool(plan))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            plan = None
    if plan is None or not isinstance(plan, dict):
        plan = tool_build_plan(question)
    plan_obj = QueryPlan.model_validate(plan)
    schema = schema_str(QueryPlan)
    user = QUERY_REFINE_USER.format(
        question=question,
        filters_json=json.dumps(plan_obj.model_dump(), ensure_ascii=False),
        schema=schema,
    )
    raw = call_llm_json(
        model=model,
        system=QUERY_REFINE_SYSTEM,
        user=user,
        schema=schema,
        max_tokens=300,
        temperature=0.0,
    )
    return QueryPlan.model_validate(raw).model_dump()


@tool("select_threads")
def tool_select_threads(question: str, threads: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Select relevant threads from candidates for the question."""
    logger.info("tool_select_threads: question=%s threads=%s", question, 0 if not threads else len(threads))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not threads:
        return []
    candidates = []
    for t in threads:
        candidates.append({
            "thread_key": t.get("thread_key") or t.get("thread_ref") or "",
            "thread_ref": t.get("thread_ref") or "",
            "topic": t.get("topic") or "",
            "counterparty": t.get("counterparty") or "",
            "handler_name": t.get("handler_name") or "",
            "customer_name": t.get("customer_name") or "",
            "thread_summary": t.get("thread_summary") or "",
        })
    schema = schema_str(ThreadSelection)
    user = THREAD_SELECT_USER.format(
        question=question,
        candidates_json=json.dumps(candidates, ensure_ascii=False),
        schema=schema,
    )
    raw = call_llm_json(
        model=model,
        system=THREAD_SELECT_SYSTEM,
        user=user,
        schema=schema,
        max_tokens=300,
        temperature=0.0,
    )
    sel = ThreadSelection.model_validate(raw)
    keys = set(k for k in sel.thread_keys if k)
    if not keys:
        return []
    return [t for t in threads if (t.get("thread_key") or t.get("thread_ref")) in keys]


@tool("retrieve")
def tool_retrieve(question: str, plan: Optional[Any] = None) -> Dict[str, Any]:
    """Retrieve threads, details, and summaries using the plan."""
    logger.info("tool_retrieve: question=%s has_plan=%s", question, bool(plan))
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            plan = None
    if plan is None or not isinstance(plan, dict):
        plan = tool_build_plan(question)
    plan_obj = QueryPlan.model_validate(plan)
    search_q = plan_obj.search_query or question

    threads = search_threads(plan_obj.thread_filter, limit=10)
    sem_threads = semantic_search_threads(search_q, limit=8, filter_spec=plan_obj.thread_filter) if search_q else []
    bm_threads = bm25_search_threads(search_q, limit=8, filter_spec=plan_obj.thread_filter) if search_q else []
    hyb_threads = hybrid_search_threads(search_q, limit=8, filter_spec=plan_obj.thread_filter) if search_q else []

    def _key(t: Dict[str, Any]) -> Optional[str]:
        return t.get("thread_key") or t.get("thread_ref")

    merged = { _key(t): t for t in threads if _key(t) }
    for bucket in (sem_threads, bm_threads, hyb_threads):
        for t in bucket:
            tkey = _key(t)
            if tkey and tkey not in merged:
                merged[tkey] = t
    threads = list(merged.values())[:10]

    q = question.lower()
    wants_detail = any(k in q for k in ["why", "details", "latest", "messages", "what did they say", "show thread"])
    wants_summary = "summary" in q or "summaries" in q

    details = []
    if wants_detail and len(threads) == 1:
        tkey = threads[0].get("thread_key") or threads[0].get("thread_ref")
        if tkey:
            detail = get_thread_detail(tkey)
            if detail:
                details = [detail]
    elif question:
        details = semantic_search_details(search_q, limit=8, filter_spec=plan_obj.detail_filter)
        details += bm25_search_details(search_q, limit=8, filter_spec=plan_obj.detail_filter)
        details += hybrid_search_details(search_q, limit=8, filter_spec=plan_obj.detail_filter)

    if details and threads:
        tkeys = {t.get("thread_key") or t.get("thread_ref") for t in threads if (t.get("thread_key") or t.get("thread_ref"))}
        details = [d for d in details if (d.get("thread_key") or d.get("thread_ref")) in tkeys]

    summaries = []
    if wants_summary:
        summaries = get_daily_summaries(plan_obj.summary_filter, limit=5)
        if search_q:
            summaries += bm25_search_summaries(search_q, limit=5)
            summaries += hybrid_search_summaries(search_q, limit=5)

    return {"threads": threads, "details": details, "summaries": summaries}


@tool("final_answer")
def tool_final_answer(question: str, retrieved: Optional[Any] = None) -> str:
    """Generate the final answer grounded in retrieved data."""
    logger.info("tool_final_answer: question=%s has_retrieved=%s", question, bool(retrieved))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if isinstance(retrieved, str):
        try:
            retrieved = json.loads(retrieved)
        except Exception:
            retrieved = None
    if retrieved is None or not isinstance(retrieved, dict):
        retrieved = tool_retrieve(question, plan=None)
    payload = json.dumps(retrieved, indent=2, ensure_ascii=False)
    user = AGENT_ANSWER_USER.format(question=question, retrieved_json=payload)
    ans = call_llm(model=model, system=AGENT_ANSWER_SYSTEM, user=user, max_tokens=700, temperature=0.2)
    return ans.strip()


class LangChainAgent:
    def __init__(self, max_steps: int = 6, max_refine: int = 2):
        self.max_steps = max_steps
        self.max_refine = max_refine
        self.tools = [tool_build_plan, tool_retrieve, tool_refine, tool_select_threads, tool_final_answer]
        self._executors: Dict[str, RunnableWithMessageHistory] = {}

    def _history_factory(self, session_id: str) -> MongoDBChatMessageHistory:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        mongo_db = os.getenv("MONGO_DB", "email_classifier")
        mongo_coll = os.getenv("MONGO_COLLECTION", "chat_history")
        return MongoDBChatMessageHistory(
            connection_string=mongo_uri,
            database_name=mongo_db,
            collection_name=mongo_coll,
            session_id=session_id,
        )

    def create_agent(self, model: str) -> RunnableWithMessageHistory:
        logger.info("create_agent: model=%s", model)
        if model in self._executors:
            return self._executors[model]
        timeout_s = int(os.getenv("LANGCHAIN_REQUEST_TIMEOUT", "60"))
        llm = ChatOpenAI(model=model, temperature=0.0, timeout=timeout_s)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a tool-using agent. Use tools to plan, retrieve, refine, and answer. If you have enough information, call final_answer."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, max_iterations=self.max_steps, verbose=False)
        runnable = RunnableWithMessageHistory(
            executor,
            self._history_factory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self._executors[model] = runnable
        return runnable

    def run(self, question: str, model: str, session_id: Optional[str] = None) -> str:
        logger.info("run: question=%s model=%s session_id=%s", question, model, session_id)
        try:
            executor = self.create_agent(model)
            retries = int(os.getenv("LANGCHAIN_RETRIES", "2"))
            last_err: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    cfg = {"configurable": {"session_id": session_id or "default"}}
                    out = executor.invoke({"input": question}, config=cfg)
                    return (out.get("output") or "").strip()
                except Exception as e:
                    last_err = e
                    logger.exception("LangChain invoke failed attempt=%s: %s", attempt + 1, e)
            raise last_err or RuntimeError("LangChain invoke failed")
        except Exception as e:
            # Best-effort fallback: return a safe message without crashing the API
            logger.exception("LangChain agent failed: %s", e)
            return f"LangChain agent error: {e}"
