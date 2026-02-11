from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from weaviate_service.weaviate_client import get_client
from weaviate_service.weaviate_service import THREAD_CLASS, DETAIL_CLASS, DAILY_SUMMARY_CLASS
from weaviate.classes.query import Filter

logger = logging.getLogger("email_classifier.agent_tools")

PRIORITY_ORDER = {"P0":0, "P1":1, "P2":2, "P3":3}

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
    prop = path[0]
    if allowed_props is not None and prop not in allowed_props:
        return None
    f = Filter.by_property(prop)
    value = None
    if "valueText" in spec:
        value = spec.get("valueText")
    elif "valueNumber" in spec:
        value = spec.get("valueNumber")
    elif "valueBoolean" in spec:
        value = spec.get("valueBoolean")
    if value is None or not op:
        return None

    op_map = {
        "Equal": "equal",
        "Like": "like",
        "GreaterThan": "greater_than",
        "GreaterThanEqual": "greater_than_equal",
        "LessThan": "less_than",
        "LessThanEqual": "less_than_equal",
    }
    method = op_map.get(op)
    if not method:
        return None
    fn = getattr(f, method, None)
    if not fn and method == "greater_than_equal":
        fn = getattr(f, "greater_than", None)
    if not fn and method == "less_than_equal":
        fn = getattr(f, "less_than", None)
    if not fn and method == "like":
        fn = getattr(f, "equal", None)
    return fn(value) if fn else None

def search_threads(filter_spec: Optional[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Hybrid: deterministic structured filtering + (optionally) later semantic.
    For MVP, we do structured filtering and return the most relevant.
    """
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)

        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("search_threads: filter=%s", filter_spec)
        res = col.query.fetch_objects(limit=limit, filters=flt) if flt else col.query.fetch_objects(limit=limit)
        objs = [o.properties for o in res.objects]
        objs.sort(key=lambda p: (_best_rank(p.get("priority_best","P3")), p.get("thread_ref","")))
        return objs
    except Exception as e:
        logger.exception("search_threads failed: %s", e)
        return []
    finally:
        client.close()

def semantic_search_threads(query: str, limit: int = 10, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("semantic_search_threads: query=%s filter=%s", query, filter_spec)
        res = col.query.near_text(query=query, limit=limit, filters=flt) if flt else col.query.near_text(query=query, limit=limit)
        out = [o.properties for o in res.objects]
        return out
    except Exception as e:
        logger.exception("semantic_search_threads failed: %s", e)
        return []
    finally:
        client.close()

def bm25_search_threads(query: str, limit: int = 10, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("bm25_search_threads: query=%s filter=%s", query, filter_spec)
        res = col.query.bm25(query=query, limit=limit, filters=flt) if flt else col.query.bm25(query=query, limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("bm25_search_threads failed: %s", e)
        return []
    finally:
        client.close()

def hybrid_search_threads(query: str, limit: int = 10, alpha: float = 0.5, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(THREAD_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=THREAD_PROPS)
        logger.info("hybrid_search_threads: query=%s filter=%s", query, filter_spec)
        res = col.query.hybrid(query=query, alpha=alpha, limit=limit, filters=flt) if flt else col.query.hybrid(query=query, alpha=alpha, limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("hybrid_search_threads failed: %s", e)
        return []
    finally:
        client.close()

def get_thread_detail(thread_key: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)

        res = col.query.fetch_objects(limit=200)
        for o in res.objects:
            p = o.properties
            if (p.get("thread_key","").lower() == thread_key.lower()) or (p.get("thread_ref","").lower() == thread_key.lower()):
                return p
        return None
    except Exception as e:
        logger.exception("get_thread_detail failed: %s", e)
        return None
    finally:
        client.close()

def semantic_search_details(query: str, limit: int = 8, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=DETAIL_PROPS)
        logger.info("semantic_search_details: query=%s filter=%s", query, filter_spec)
        res = col.query.near_text(query=query, limit=limit, filters=flt) if flt else col.query.near_text(query=query, limit=limit)
        out = [o.properties for o in res.objects]
        return out
    except Exception as e:
        logger.exception("semantic_search_details failed: %s", e)
        return []
    finally:
        client.close()

def bm25_search_details(query: str, limit: int = 8, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=DETAIL_PROPS)
        logger.info("bm25_search_details: query=%s filter=%s", query, filter_spec)
        res = col.query.bm25(query=query, limit=limit, filters=flt) if flt else col.query.bm25(query=query, limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("bm25_search_details failed: %s", e)
        return []
    finally:
        client.close()

def hybrid_search_details(query: str, limit: int = 8, alpha: float = 0.5, filter_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DETAIL_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=DETAIL_PROPS)
        logger.info("hybrid_search_details: query=%s filter=%s", query, filter_spec)
        res = col.query.hybrid(query=query, alpha=alpha, limit=limit, filters=flt) if flt else col.query.hybrid(query=query, alpha=alpha, limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("hybrid_search_details failed: %s", e)
        return []
    finally:
        client.close()

def get_daily_summaries(filter_spec: Optional[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DAILY_SUMMARY_CLASS)
        flt = _filter_from_spec(filter_spec, allowed_props=SUMMARY_PROPS)
        logger.info("get_daily_summaries: filter=%s", filter_spec)
        res = col.query.fetch_objects(limit=limit, filters=flt) if flt else col.query.fetch_objects(limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("get_daily_summaries failed: %s", e)
        return []
    finally:
        client.close()

def bm25_search_summaries(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DAILY_SUMMARY_CLASS)
        res = col.query.bm25(query=query, limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("bm25_search_summaries failed: %s", e)
        return []
    finally:
        client.close()

def hybrid_search_summaries(query: str, limit: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        client.connect()
        col = client.collections.get(DAILY_SUMMARY_CLASS)
        res = col.query.hybrid(query=query, alpha=alpha, limit=limit)
        return [o.properties for o in res.objects]
    except Exception as e:
        logger.exception("hybrid_search_summaries failed: %s", e)
        return []
    finally:
        client.close()
