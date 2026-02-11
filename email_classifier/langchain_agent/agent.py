from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from email_classifier.chat_service.agent_tools import (
    bm25_search_threads,
    get_collection_properties,
    get_daily_summaries,
    hybrid_search_summaries,
    hybrid_search_threads,
    semantic_search_threads,
)
from email_classifier.langchain_agent.mongo_history import MongoChatHistory
from email_classifier.shared.config import MAX_DETAIL_CHARS
from email_classifier.shared.llm import call_llm, call_llm_json, schema_str
from email_classifier.shared.models import QueryPlan
from email_classifier.shared.prompts import (
    AGENT_ANSWER_SYSTEM,
    AGENT_ANSWER_USER,
    QUERY_REFINE_SYSTEM,
    QUERY_REFINE_USER,
    QUERY_TO_FILTER_SYSTEM,
    QUERY_TO_FILTER_USER,
)

logger = logging.getLogger("email_classifier.langchain_agent")


def _is_summary_query(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ("summary", "summaries", "daily summary", "day summary"))


def _dedupe_threads(items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for t in items:
        key = t.get("thread_key") or t.get("thread_ref")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _thread_view(t: dict[str, Any]) -> dict[str, Any]:
    out = {
        "thread_key": t.get("thread_key"),
        "thread_ref": t.get("thread_ref"),
        "email_type": t.get("email_type"),
        "priority_best": t.get("priority_best"),
        "topic": t.get("topic"),
        "counterparty": t.get("counterparty"),
        "handler_name": t.get("handler_name"),
        "customer_name": t.get("customer_name"),
        "thread_summary": t.get("thread_summary"),
        "urgency_reason": t.get("urgency_reason"),
        "latest_sent_at": t.get("latest_sent_at"),
    }
    full_text = t.get("full_text")
    if isinstance(full_text, str) and full_text:
        out["full_text"] = full_text[:MAX_DETAIL_CHARS]
    return out


def _build_plan_sync(question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    props = get_collection_properties()
    schema = schema_str(QueryPlan)
    user = QUERY_TO_FILTER_USER.format(
        question=question,
        properties_json=json.dumps(props, ensure_ascii=False),
        schema=schema,
    )
    raw = call_llm_json(
        model=model,
        system=QUERY_TO_FILTER_SYSTEM,
        user=user,
        schema=schema,
        temperature=0.0,
    )
    logger.info("timing.build_plan_sec=%.3f", time.perf_counter() - t0)
    return QueryPlan.model_validate(raw).model_dump()


async def _build_plan(question: str) -> dict[str, Any]:
    return await asyncio.to_thread(_build_plan_sync, question)


@tool("build_plan")
async def tool_build_plan(question: str) -> dict[str, Any]:
    """Build a compact QueryPlan from the user question."""
    logger.info("tool_build_plan: question_len=%s", len(question or ""))
    return await _build_plan(question)


def _refine_plan_sync(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    props = get_collection_properties()
    schema = schema_str(QueryPlan)
    user = QUERY_REFINE_USER.format(
        question=question,
        properties_json=json.dumps(props, ensure_ascii=False),
        filters_json=json.dumps(plan, ensure_ascii=False),
        schema=schema,
    )
    raw = call_llm_json(
        model=model,
        system=QUERY_REFINE_SYSTEM,
        user=user,
        schema=schema,
        temperature=0.0,
    )
    logger.info("timing.refine_plan_sec=%.3f", time.perf_counter() - t0)
    return QueryPlan.model_validate(raw).model_dump()


async def _refine_plan(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(_refine_plan_sync, question, plan)


def _retrieve(question: str, plan: Any | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            plan = None
    if not isinstance(plan, dict):
        plan = _build_plan_sync(question)
    plan_obj = QueryPlan.model_validate(plan)
    search_q = plan_obj.search_query or question

    retrieve_limit = int(os.getenv("RETRIEVE_THREADS_LIMIT", "5"))
    # Stage 1: hybrid + bm25 (with and without filter)
    threads_unf = hybrid_search_threads(search_q, retrieve_limit, 0.7, None)
    threads_flt = hybrid_search_threads(search_q, retrieve_limit, 0.7, plan_obj.thread_filter)
    threads_unf += bm25_search_threads(search_q, retrieve_limit, None)
    threads_flt += bm25_search_threads(search_q, retrieve_limit, plan_obj.thread_filter)
    # Prefer filtered results first, then backfill from unfiltered.
    threads = _dedupe_threads(threads_flt + threads_unf, limit=retrieve_limit)

    # Stage 2 fallback: semantic only if no docs after hybrid + bm25
    if not threads:
        sem_unf = semantic_search_threads(search_q, retrieve_limit, None)
        sem_flt = semantic_search_threads(search_q, retrieve_limit, plan_obj.thread_filter)
        threads = _dedupe_threads(sem_flt + sem_unf, limit=retrieve_limit)

    # Optional refine retry only when both calls are empty (off by default for speed).
    enable_refine = os.getenv("RETRIEVE_ENABLE_REFINE", "false").lower() in {"1", "true", "yes"}
    if not threads and enable_refine:
        refined = _refine_plan_sync(question, plan_obj.model_dump())
        rplan = QueryPlan.model_validate(refined)
        refined_res = hybrid_search_threads(search_q, retrieve_limit, 0.5, rplan.thread_filter)
        refined_res += bm25_search_threads(search_q, retrieve_limit, rplan.thread_filter)
        if not refined_res:
            refined_res = semantic_search_threads(search_q, retrieve_limit, rplan.thread_filter)
        if isinstance(refined_res, list):
            threads = _dedupe_threads(refined_res, limit=retrieve_limit)

    thread_ids = []
    seen = set()
    for t in threads:
        k = t.get("thread_key") or t.get("thread_ref")
        if k and k not in seen:
            seen.add(k)
            thread_ids.append(k)
    summaries: list[dict[str, Any]] = []
    q = question.lower()
    if "summary" in q or "summaries" in q:
        s1 = get_daily_summaries(plan_obj.summary_filter, 5)
        s2 = hybrid_search_summaries(search_q, 5, 0.5)
        if isinstance(s1, list):
            summaries.extend(s1)
        if isinstance(s2, list):
            summaries.extend(s2)

    logger.info(
        "timing.retrieve_sec=%.3f threads=%s summaries=%s",
        time.perf_counter() - t0,
        len(threads),
        len(summaries),
    )
    return {"threads": threads, "summaries": summaries}


@tool("retrieve")
def tool_retrieve(question: str, plan: Any | None = None) -> dict[str, Any]:
    """Retrieve top thread candidates (with and without filters) from EmailThread."""
    logger.info("tool_retrieve: question_len=%s has_plan=%s", len(question or ""), bool(plan))
    return _retrieve(question, plan)


def _sanitize_retrieved(retrieved: dict[str, Any]) -> dict[str, Any]:
    src = dict(retrieved or {})
    threads_src = [t for t in (src.get("threads") or []) if isinstance(t, dict)]
    sums_src = [s for s in (src.get("summaries") or []) if isinstance(s, dict)]
    dedup_summaries: list[dict[str, Any]] = []
    seen_sum = set()
    for s in sums_src:
        key = (s.get("user_email_lc") or s.get("user_email") or "", s.get("day") or "")
        if key in seen_sum:
            continue
        seen_sum.add(key)
        dedup_summaries.append(s)

    return {
        "threads": [_thread_view(t) for t in threads_src[: int(os.getenv('FINAL_ANSWER_THREADS_TOP_N', '3'))]],
        "summaries": dedup_summaries[:5],
    }


async def _final_answer(question: str, retrieved: Any | None = None) -> str:
    t0 = time.perf_counter()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if isinstance(retrieved, str):
        try:
            retrieved = json.loads(retrieved)
        except Exception:
            retrieved = None
    if not isinstance(retrieved, dict):
        retrieved = _retrieve(question, None)
    payload = json.dumps(_sanitize_retrieved(retrieved), ensure_ascii=False, indent=2)
    user = AGENT_ANSWER_USER.format(question=question, retrieved_json=payload)
    final_max_tokens = int(os.getenv("FINAL_ANSWER_MAX_TOKENS", "450"))
    ans = await asyncio.to_thread(
        call_llm,
        model=model,
        system=AGENT_ANSWER_SYSTEM,
        user=user,
        max_tokens=final_max_tokens,
        temperature=0.1,
    )
    logger.info(
        "timing.final_answer_sec=%.3f payload_chars=%s max_tokens=%s",
        time.perf_counter() - t0,
        len(payload),
        final_max_tokens,
    )
    return ans.strip()


@tool("final_answer")
async def tool_final_answer(question: str, retrieved: Any | None = None) -> str:
    """Generate grounded answer from retrieved records."""
    logger.info("tool_final_answer: question_len=%s has_retrieved=%s", len(question or ""), bool(retrieved))
    return await _final_answer(question, retrieved)


class LangChainAgent:
    def __init__(self, max_steps: int = 3):
        self.max_steps = max_steps
        self.tools = [tool_build_plan, tool_retrieve, tool_final_answer]
        self._executors: dict[str, RunnableWithMessageHistory] = {}

    def _history_factory(self, session_id: str) -> MongoChatHistory:
        return MongoChatHistory(session_id=session_id)

    def create_agent(self, model: str) -> RunnableWithMessageHistory:
        if model in self._executors:
            return self._executors[model]
        timeout_s = int(os.getenv("LANGCHAIN_REQUEST_TIMEOUT", "60"))
        llm = ChatOpenAI(model=model, temperature=0.0, timeout=timeout_s)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a tool-using insurance ops agent.\n"
                "Always follow this order: build_plan -> retrieve -> final_answer.\n"
                "Do not call tools repeatedly unless a call fails.\n"
                "Never call build_plan more than once per question.\n"
                "Never call final_answer before retrieve.\n"
                "Keep responses factual and evidence-grounded.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            max_iterations=self.max_steps,
            verbose=False,
        )
        runnable = RunnableWithMessageHistory(
            executor,
            self._history_factory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self._executors[model] = runnable
        return runnable

    async def arun(self, question: str, model: str, session_id: str | None = None) -> str:
        t0 = time.perf_counter()
        logger.info("run: question_len=%s model=%s session_id=%s", len(question or ""), model, session_id)
        try:
            # Deterministic fast-path for summary requests avoids tool-loop max-iteration failures.
            if _is_summary_query(question):
                logger.info("run: summary fast-path enabled")
                plan = await asyncio.to_thread(_build_plan_sync, question)
                retrieved = await asyncio.to_thread(_retrieve, question, plan)
                answer = await _final_answer(question, retrieved)
                logger.info("timing.total_arun_sec=%.3f", time.perf_counter() - t0)
                return answer

            executor = self.create_agent(model)
            cfg = {"configurable": {"session_id": session_id or "default"}}
            out = await executor.ainvoke({"input": question}, config=cfg)
            logger.info("timing.total_arun_sec=%.3f", time.perf_counter() - t0)
            return (out.get("output") or "").strip()
        except Exception as e:
            logger.exception("LangChain agent failed: %s", e)
            return f"LangChain agent error: {e}"

    def run(self, question: str, model: str, session_id: str | None = None) -> str:
        return asyncio.run(self.arun(question, model, session_id))
