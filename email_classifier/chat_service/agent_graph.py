from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

import logging
from email_classifier.shared.models import QueryPlan, ThreadSelection
from email_classifier.shared.config import MAX_DETAIL_CHARS
from email_classifier.shared.utils import extract_claim_ref
from email_classifier.shared.llm import call_llm_json, call_llm, schema_str
from email_classifier.shared.prompts import (
    QUERY_TO_FILTER_SYSTEM, QUERY_TO_FILTER_USER,
    QUERY_REFINE_SYSTEM, QUERY_REFINE_USER,
    THREAD_SELECT_SYSTEM, THREAD_SELECT_USER,
    AGENT_ANSWER_SYSTEM, AGENT_ANSWER_USER,
)
from email_classifier.chat_service.agent_tools import (
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
    get_collection_properties,
)

logger = logging.getLogger("email_classifier.agent_graph")
class AgentState(TypedDict, total=False):
    question: str
    model: str
    plan: Dict[str, Any]
    retrieved: Dict[str, Any]
    answer: str
    attempts: int
    need_refine: bool

def node_interpret(state: AgentState) -> AgentState:
    model = state["model"]
    logger.info("node_interpret: question_len=%s", len(state.get("question") or ""))
    schema = schema_str(QueryPlan)
    user = QUERY_TO_FILTER_USER.format(
        question=state["question"],
        properties_json=json.dumps(get_collection_properties(), ensure_ascii=False),
        schema=schema,
    )
    raw = call_llm_json(
        model=model,
        system=QUERY_TO_FILTER_SYSTEM,
        user=user,
        schema=schema,
        temperature=0.0,
    )
    plan = QueryPlan.model_validate(raw)
    # Hard override for explicit claim refs to avoid LLM filter misses
    claim_ref = extract_claim_ref(state.get("question"), state.get("question"))
    if claim_ref:
        plan.thread_filter = {
            "path": ["thread_ref"],
            "operator": "Equal",
            "valueText": claim_ref,
        }
        logger.info("node_interpret: override thread_filter for claim_ref=%s", claim_ref)
    return {"plan": plan.model_dump(), "attempts": 0, "need_refine": False}

def _refine_filters(question: str, plan: QueryPlan, model: str) -> QueryPlan:
    logger.info("refine_filters: question_len=%s", len(question or ""))
    schema = schema_str(QueryPlan)
    user = QUERY_REFINE_USER.format(
        question=question,
        properties_json=json.dumps(get_collection_properties(), ensure_ascii=False),
        filters_json=json.dumps(plan.model_dump(), ensure_ascii=False),
        schema=schema,
    )
    raw = call_llm_json(
        model=model,
        system=QUERY_REFINE_SYSTEM,
        user=user,
        schema=schema,
        temperature=0.0,
    )
    return QueryPlan.model_validate(raw)

def _select_relevant_threads(question: str, threads: List[Dict[str, Any]], model: str) -> tuple[List[Dict[str, Any]], bool]:
    if not threads:
        return threads, False
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
        temperature=0.0,
    )
    sel = ThreadSelection.model_validate(raw)
    keys = set(k for k in sel.thread_keys if k)
    if not keys:
        return [], True
    return [t for t in threads if (t.get("thread_key") or t.get("thread_ref")) in keys], False

def node_retrieve(state: AgentState) -> AgentState:
    plan = QueryPlan.model_validate(state.get("plan", {}))
    question = state.get("question") or ""
    model = state.get("model") or "gpt-4o-mini"
    attempts = int(state.get("attempts", 0))
    logger.info("node_retrieve: question_len=%s attempts=%s", len(question or ""), attempts)
    search_q = plan.search_query or question

    # Step 1: retrieve without filters
    base_threads = search_threads(None, limit=10)
    base_threads += semantic_search_threads(search_q, limit=8, filter_spec=None) if search_q else []
    base_threads += bm25_search_threads(search_q, limit=8, filter_spec=None) if search_q else []
    base_threads += hybrid_search_threads(search_q, limit=8, filter_spec=None) if search_q else []

    # merge by thread_key
    def _key(t: Dict[str, Any]) -> Optional[str]:
        return t.get("thread_key") or t.get("thread_ref")

    merged = { _key(t): t for t in base_threads if _key(t) }
    threads = list(merged.values())[:10]

    # Step 2: LLM selects relevant docs from unfiltered pool
    selected = threads
    if question and len(threads) > 3:
        selected, empty_sel = _select_relevant_threads(question, threads, model)
        selected = selected if not empty_sel else threads

    # Step 3: apply LLM-generated filters directly in Weaviate
    filtered_threads = []
    if plan.thread_filter:
        filtered_threads = search_threads(plan.thread_filter, limit=10)
        filtered_threads += semantic_search_threads(search_q, limit=8, filter_spec=plan.thread_filter) if search_q else []
        filtered_threads += bm25_search_threads(search_q, limit=8, filter_spec=plan.thread_filter) if search_q else []
        filtered_threads += hybrid_search_threads(search_q, limit=8, filter_spec=plan.thread_filter) if search_q else []

    merged = { _key(t): t for t in selected if _key(t) }
    for t in filtered_threads:
        tkey = _key(t)
        if tkey and tkey not in merged:
            merged[tkey] = t
    threads = list(merged.values())[:10]

    # If user wants details and we have a single thread, fetch messages
    q = question.lower()
    wants_detail = bool(plan.need_detail) if plan.need_detail is not None else any(
        k in q for k in ["why", "details", "latest", "messages", "what did they say", "show thread"]
    )
    wants_summary = "summary" in q or "summaries" in q
    details = []
    # Step 4: fetch full thread content for all merged thread ids
    thread_ids = []
    seen = set()
    for t in threads:
        tkey = t.get("thread_key") or t.get("thread_ref")
        if tkey and tkey not in seen:
            seen.add(tkey)
            thread_ids.append(tkey)
    for tkey in thread_ids:
        detail = get_thread_detail(tkey)
        if detail:
            details.append(detail)

    # Keep only details for threads in the merged set when possible
    if details and threads:
        tkeys = {t.get("thread_key") or t.get("thread_ref") for t in threads if (t.get("thread_key") or t.get("thread_ref"))}
        details = [d for d in details if (d.get("thread_key") or d.get("thread_ref")) in tkeys]

    summaries = []
    if wants_summary:
        summaries = get_daily_summaries(plan.summary_filter, limit=5)
        if search_q:
            summaries += bm25_search_summaries(search_q, limit=5)
            summaries += hybrid_search_summaries(search_q, limit=5)

    empty_all = not threads and not details and not summaries
    need_refine = bool(search_q) and (empty_all)
    return {
        "retrieved": {"threads": threads, "details": details, "summaries": summaries},
        "need_refine": need_refine,
        "attempts": attempts,
    }

def node_refine(state: AgentState) -> AgentState:
    plan = QueryPlan.model_validate(state.get("plan", {}))
    question = state.get("question") or ""
    model = state.get("model") or "gpt-4o-mini"
    attempts = int(state.get("attempts", 0)) + 1
    logger.info("node_refine: question_len=%s attempts=%s", len(question or ""), attempts)
    plan2 = _refine_filters(question, plan, model)
    return {"plan": plan2.model_dump(), "attempts": attempts, "need_refine": False}

def node_generate_answer(state: AgentState) -> AgentState:
    model = state["model"]
    payload = json.dumps(_sanitize_retrieved(state.get("retrieved", {})), indent=2, ensure_ascii=False)
    user = AGENT_ANSWER_USER.format(question=state["question"], retrieved_json=payload)
    ans = call_llm(model=model, system=AGENT_ANSWER_SYSTEM, user=user, temperature=0.2)
    return {"answer": ans.strip()}

def _sanitize_retrieved(retrieved: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(retrieved or {})
    details = out.get("details") or []
    sanitized_details = []
    for d in details:
        if not isinstance(d, dict):
            continue
        d = dict(d)
        d.pop("messages_json", None)
        if isinstance(d.get("full_text"), str):
            d["full_text"] = d["full_text"][:MAX_DETAIL_CHARS]
        sanitized_details.append(d)
    out["details"] = sanitized_details
    return out

def _filter_has_thread_ref(filter_spec: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(filter_spec, dict):
        return False
    if "path" in filter_spec:
        path = filter_spec.get("path") or []
        return len(path) > 0 and path[0] == "thread_ref"
    if "operator" in filter_spec and "operands" in filter_spec:
        for op in filter_spec.get("operands") or []:
            if _filter_has_thread_ref(op):
                return True
    if "conditions" in filter_spec or "groups" in filter_spec:
        for cond in filter_spec.get("conditions") or []:
            if isinstance(cond, dict) and cond.get("property") == "thread_ref":
                return True
        for grp in filter_spec.get("groups") or []:
            if _filter_has_thread_ref(grp):
                return True
    return False

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("interpret", node_interpret)
    g.add_node("retrieve", node_retrieve)
    g.add_node("refine", node_refine)
    g.add_node("generate_answer", node_generate_answer)
    g.set_entry_point("interpret")
    g.add_edge("interpret", "retrieve")
    def _should_refine(state: AgentState) -> str:
        if state.get("need_refine") and int(state.get("attempts", 0)) < 2:
            return "refine"
        return "generate_answer"
    g.add_conditional_edges("retrieve", _should_refine, {"refine": "refine", "generate_answer": "generate_answer"})
    g.add_edge("refine", "retrieve")
    g.add_edge("generate_answer", END)
    return g.compile()
