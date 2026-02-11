from __future__ import annotations
import logging
import json
import re
import os
from typing import Any, Dict, List, Optional
from email_classifier.weaviate_service.weaviate_client import get_fresh_client
from email_classifier.weaviate_service.weaviate_service import THREAD_CLASS, DETAIL_CLASS, DAILY_SUMMARY_CLASS
from email_classifier.shared.config import THREAD_FETCH_LIMIT, DETAIL_FETCH_LIMIT, SUMMARY_FETCH_LIMIT
from weaviate.classes.query import Filter, Rerank

logger = logging.getLogger("email_classifier.agent_tools")

PRIORITY_ORDER = {"P0":0, "P1":1, "P2":2, "P3":3}

_STOPWORDS_FALLBACK = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","had",
    "he","her","hers","him","his","i","in","is","it","its","me","my","of","on","or",
    "our","ours","she","that","the","their","theirs","them","they","this","to","was",
    "we","were","what","when","where","which","who","why","with","you","your","yours",
    "please","kindly","could","would","should","can","will","just","need","needs",
    "show","tell","get","find","list","any","all","under","about","regarding"
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9'-]+")

def _clean_query(query: str) -> str:
    if not query:
        return ""
    tokens = _TOKEN_RE.findall(query.lower())
    kept = [t for t in tokens if t not in _STOPWORDS_FALLBACK and len(t) > 1]
    return " ".join(kept) if kept else query

def get_collection_properties() -> Dict[str, List[Dict[str, Any]]]:
    client = get_fresh_client()
    try:
        client.connect()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for cname in (THREAD_CLASS, DETAIL_CLASS, DAILY_SUMMARY_CLASS):
            props: List[Dict[str, Any]] = []
            try:
                col = client.collections.get(cname)
                cfg = col.config.get()
                for p in cfg.properties:
                    props.append({"name": p.name, "data_type": str(getattr(p, "data_type", ""))})
            except Exception:
                props = []
            out[cname] = props
        return out
    finally:
        client.close()


def _rerank_enabled() -> bool:
    return os.getenv("WEAVIATE_RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}

_RERANK_AVAILABLE: Optional[bool] = None


def _run_with_optional_rerank(search_fn, *, query_text: str, rerank_prop: str, **kwargs):
    """
    Run a Weaviate search with server-side rerank and fall back transparently
    if rerank is not available/configured in the target Weaviate instance.
    """
    global _RERANK_AVAILABLE
    if _rerank_enabled() and query_text and _RERANK_AVAILABLE is not False:
        try:
            out = search_fn(
                query=query_text,
                rerank=Rerank(prop=rerank_prop, query=query_text),
                **kwargs,
            )
            _RERANK_AVAILABLE = True
            return out
        except Exception as e:
            _RERANK_AVAILABLE = False
            logger.warning("rerank unavailable; fallback to base query: %s", e)
    return search_fn(query=query_text, **kwargs)

THREAD_PROPS = {
    "thread_id","thread_key","thread_ref","email_type","topic","topic_lc","category",
    "priority_best","action_required","actions_json","counterparty","counterparty_lc",
    "entities_json","thread_summary","latest_message","participants_text","latest_sent_at",
    "confidence","archive_recommendation","urgency_reason","missing_info_json","handler_name",
    "handler_name_lc","handler_email","customer_name","customer_name_lc","customer_email",
    "user_email","user_email_lc",
}

DETAIL_PROPS = {
    "thread_id","thread_key","thread_ref","user_email","user_email_lc","full_text",
    "participants_text","messages_json","latest_sent_at","earliest_sent_at","message_count",
}

SUMMARY_PROPS = {
    "day","user_email","user_email_lc","summary_md","total_threads","action_required",
    "informational","irrelevant","created_at",
}

def _best_rank(priority: str) -> int:
    return PRIORITY_ORDER.get(priority or "P3", 3)

def _filter_from_spec(spec: Optional[Dict[str, Any]], allowed_props: Optional[set[str]] = None) -> Optional[Filter]:
    if not spec:
        return None
    # LLM-friendly JSON format:
    # {"op":"and|or","conditions":[...],"groups":[...]}
    if "op" in spec or "conditions" in spec or "groups" in spec:
        op = str(spec.get("op", "and")).lower()
        built: List[Filter] = []
        for cond in spec.get("conditions", []) or []:
            if not isinstance(cond, dict):
                continue
            prop = cond.get("property")
            if not prop:
                continue
            lc_prop = f"{prop}_lc"
            if allowed_props is not None and prop not in allowed_props:
                if lc_prop not in allowed_props:
                    continue
                prop = lc_prop
            val = cond.get("value")
            typ = str(cond.get("type", "string")).lower()
            oper = str(cond.get("operator", "equal")).lower()
            if typ in ("number", "int", "float"):
                try:
                    val = float(val)
                except Exception:
                    continue
            elif typ in ("bool", "boolean"):
                val = str(val).lower() in {"true", "1", "yes"}
            if isinstance(val, str) and prop.endswith("_lc"):
                val = val.lower()
            f = Filter.by_property(prop)
            if oper in ("equal", "eq"):
                built.append(f.equal(val))
            elif oper in ("like", "contains"):
                if isinstance(val, str) and "*" not in val and "?" not in val:
                    val = f"*{val}*"
                built.append(f.like(val))
            elif oper in ("gt", "greaterthan"):
                built.append(f.greater_than(val))
            elif oper in ("gte", "greaterthanequal"):
                built.append(f.greater_than(val))
            elif oper in ("lt", "lessthan"):
                built.append(f.less_than(val))
            elif oper in ("lte", "lessthanequal"):
                built.append(f.less_than(val))
            elif oper in ("notequal", "ne"):
                built.append(f.not_equal(val))
        for grp in spec.get("groups", []) or []:
            sub = _filter_from_spec(grp, allowed_props)
            if sub is not None:
                built.append(sub)
        if not built:
            return None
        flt = built[0]
        for b in built[1:]:
            flt = (flt | b) if op == "or" else (flt & b)
        return flt

    op = spec.get("operator")
    operands = spec.get("operands")
    if op and isinstance(operands, list):
        built = [_filter_from_spec(o, allowed_props=allowed_props) for o in operands]
        built = [b for b in built if b is not None]
        if not built:
            return None
        if op.lower() == "and":
            flt = built[0]
            for b in built[1:]:
                flt = flt & b
            return flt
        if op.lower() == "or":
            flt = built[0]
            for b in built[1:]:
                flt = flt | b
            return flt
        return None

    path = spec.get("path") or []
    if not path:
        return None
    value = None
    if "valueText" in spec:
        value = spec.get("valueText")
    elif "valueNumber" in spec:
        value = spec.get("valueNumber")
    elif "valueBoolean" in spec:
        value = spec.get("valueBoolean")
    if value is None or not op:
        return None

    prop = path[0]
    lc_prop = f"{prop}_lc"
    if allowed_props is not None and prop not in allowed_props:
        if lc_prop not in allowed_props:
            return None
        prop = lc_prop
    # Prefer lowercase variant for case-insensitive string matching
    if isinstance(value, str) and allowed_props is not None and lc_prop in allowed_props and prop != lc_prop:
        prop = lc_prop

    f = Filter.by_property(prop)

    # Normalize operator and value for Weaviate v4 Filter
    op_norm = str(op).strip()
    op_map = {
        "equal": "equal",
        "like": "like",
        "contains": "like",
        "greaterthan": "greater_than",
        "greaterthanequal": "greater_than_equal",
        "lessthan": "less_than",
        "lessthanequal": "less_than_equal",
    }
    method = op_map.get(op_norm.lower())
    if not method:
        return None
    if isinstance(value, str):
        if prop.endswith("_lc"):
            value = value.lower()
        if method == "like":
            # Ensure wildcard for partial matching if not provided
            if "*" not in value and "?" not in value:
                value = f"*{value}*"
    fn = getattr(f, method, None)
    if not fn and method == "greater_than_equal":
        fn = getattr(f, "greater_than", None)
    if not fn and method == "less_than_equal":
        fn = getattr(f, "less_than", None)
    if not fn and method == "like":
        fn = getattr(f, "equal", None)
    return fn(value) if fn else None

def search_threads(filter_spec: Optional[Dict[str, Any]], limit: int = THREAD_FETCH_LIMIT) -> List[Dict[str, Any]]:
    """
    Hybrid: deterministic structured filtering + (optionally) later semantic.
    For MVP, we do structured filtering and return the most relevant.
    """
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)

        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("search_threads: filter=%s", _filter_summary(filter_spec))
        res = col.query.fetch_objects(limit=limit, filters=flt) if flt else col.query.fetch_objects(limit=limit)
        objs = [o.properties for o in res.objects]
        logger.info("search_threads: fetched=%s", len(objs))
        objs.sort(key=lambda p: (_best_rank(p.get("priority_best","P3")), p.get("thread_ref","")))
        return objs
    except Exception as e:
        logger.exception("search_threads failed: %s", e)
        return []
    finally:
        client.close()

def semantic_search_threads(query: str, limit: int = THREAD_FETCH_LIMIT, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    query = _clean_query(query)
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("semantic_search_threads: query_len=%s filter=%s", len(query or ""), _filter_summary(filter_spec))
        logger.info("semantic_search_threads: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.near_text,
            query_text=query,
            rerank_prop="thread_summary",
            limit=limit,
            filters=flt,
        ) if flt else _run_with_optional_rerank(
            col.query.near_text,
            query_text=query,
            rerank_prop="thread_summary",
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        logger.info("semantic_search_threads: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("semantic_search_threads failed: %s", e)
        return []
    finally:
        client.close()

def bm25_search_threads(query: str, limit: int = THREAD_FETCH_LIMIT, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("bm25_search_threads: query_len=%s filter=%s", len(query or ""), _filter_summary(filter_spec))
        logger.info("bm25_search_threads: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.bm25,
            query_text=query,
            rerank_prop="thread_summary",
            limit=limit,
            filters=flt,
        ) if flt else _run_with_optional_rerank(
            col.query.bm25,
            query_text=query,
            rerank_prop="thread_summary",
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        logger.info("bm25_search_threads: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("bm25_search_threads failed: %s", e)
        return []
    finally:
        client.close()

def hybrid_search_threads(query: str, limit: int = THREAD_FETCH_LIMIT, alpha: float = 0.5, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    query = _clean_query(query)
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("hybrid_search_threads: query_len=%s filter=%s", len(query or ""), _filter_summary(filter_spec))
        logger.info("hybrid_search_threads: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.hybrid,
            query_text=query,
            rerank_prop="thread_summary",
            alpha=alpha,
            limit=limit,
            filters=flt,
        ) if flt else _run_with_optional_rerank(
            col.query.hybrid,
            query_text=query,
            rerank_prop="thread_summary",
            alpha=alpha,
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        logger.info("hybrid_search_threads: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("hybrid_search_threads failed: %s", e)
        return []
    finally:
        client.close()

def get_thread_detail(thread_key: str) -> Optional[Dict[str, Any]]:
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = Filter.by_property("thread_key").equal(thread_key)
        res = col.query.fetch_objects(limit=1, filters=flt)
        if res.objects:
            return res.objects[0].properties
        return None
    except Exception as e:
        logger.exception("get_thread_detail failed: %s", e)
        return None
    finally:
        client.close()

def semantic_search_details(query: str, limit: int = DETAIL_FETCH_LIMIT, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    query = _clean_query(query)
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=DETAIL_PROPS)
        logger.info("semantic_search_details: query_len=%s filter=%s", len(query or ""), _filter_summary(filter_spec))
        logger.info("semantic_search_details: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.near_text,
            query_text=query,
            rerank_prop="full_text",
            limit=limit,
            filters=flt,
        ) if flt else _run_with_optional_rerank(
            col.query.near_text,
            query_text=query,
            rerank_prop="full_text",
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        return out
    except Exception as e:
        logger.exception("semantic_search_details failed: %s", e)
        return []
    finally:
        client.close()

def bm25_search_details(query: str, limit: int = DETAIL_FETCH_LIMIT, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=DETAIL_PROPS)
        logger.info("bm25_search_details: query_len=%s filter=%s", len(query or ""), _filter_summary(filter_spec))
        logger.info("bm25_search_details: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.bm25,
            query_text=query,
            rerank_prop="full_text",
            limit=limit,
            filters=flt,
        ) if flt else _run_with_optional_rerank(
            col.query.bm25,
            query_text=query,
            rerank_prop="full_text",
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        logger.info("semantic_search_details: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("bm25_search_details failed: %s", e)
        return []
    finally:
        client.close()

def hybrid_search_details(query: str, limit: int = DETAIL_FETCH_LIMIT, alpha: float = 0.5, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    query = _clean_query(query)
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=DETAIL_PROPS)
        logger.info("hybrid_search_details: query_len=%s filter=%s", len(query or ""), _filter_summary(filter_spec))
        logger.info("hybrid_search_details: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.hybrid,
            query_text=query,
            rerank_prop="full_text",
            alpha=alpha,
            limit=limit,
            filters=flt,
        ) if flt else _run_with_optional_rerank(
            col.query.hybrid,
            query_text=query,
            rerank_prop="full_text",
            alpha=alpha,
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        logger.info("bm25_search_details: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("hybrid_search_details failed: %s", e)
        return []
    finally:
        client.close()

def get_daily_summaries(filter_spec: Optional[Dict[str, Any]], limit: int = SUMMARY_FETCH_LIMIT) -> List[Dict[str, Any]]:
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DAILY_SUMMARY_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=SUMMARY_PROPS)
        logger.info("get_daily_summaries: filter=%s", _filter_summary(filter_spec))
        res = col.query.fetch_objects(limit=limit, filters=flt) if flt else col.query.fetch_objects(limit=limit)
        out = [o.properties for o in res.objects]
        logger.info("hybrid_search_details: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("get_daily_summaries failed: %s", e)
        return []
    finally:
        client.close()

def bm25_search_summaries(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DAILY_SUMMARY_CLASS)
        logger.info("bm25_search_summaries: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.bm25,
            query_text=query,
            rerank_prop="summary_md",
            limit=limit,
        )
        out = [o.properties for o in res.objects]
        logger.info("get_daily_summaries: fetched=%s", len(out))
        return out
    except Exception as e:
        logger.exception("bm25_search_summaries failed: %s", e)
        return []
    finally:
        client.close()

def hybrid_search_summaries(query: str, limit: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
    client = get_fresh_client()
    try:
        client.connect()
        col = client.collections.get(DAILY_SUMMARY_CLASS)
        logger.info("hybrid_search_summaries: rerank=%s", _rerank_enabled())
        res = _run_with_optional_rerank(
            col.query.hybrid,
            query_text=query,
            rerank_prop="summary_md",
            alpha=alpha,
            limit=limit,
        )
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("hybrid_search_summaries failed: %s", e)
        return []
    finally:
        client.close()
def _filter_summary(filter_spec: Optional[Dict[str, Any]]) -> str:
    if not filter_spec:
        return "none"
    def _strip(spec: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in spec.items() if k in ("path","operator","operands")}
        if "operands" in out and isinstance(out["operands"], list):
            out["operands"] = [_strip(o) for o in out["operands"] if isinstance(o, dict)]
        return out
    try:
        return json.dumps(_strip(filter_spec), ensure_ascii=False)
    except Exception:
        return "unserializable"
