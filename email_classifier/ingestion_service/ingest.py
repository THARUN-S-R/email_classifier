from __future__ import annotations

import argparse, os, json, logging, hashlib, uuid
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

from email_classifier.shared.models import EmailThread, ThreadTriage
from email_classifier.shared.utils import (
    ensure_dir, load_json, write_text, write_json, append_jsonl,
    extract_claim_ref, CLAIM_REF_RE, sender_domain, redact_for_llm, redact_for_index, raw_store,
    extract_signature_name, extract_salutation_name, name_from_email,
    parse_datetime
)
from email_classifier.shared.llm import call_llm_json_model, schema_str
from email_classifier.shared.prompts import THREAD_TRIAGE_SYSTEM, THREAD_TRIAGE_USER, FEW_SHOT_EXAMPLES
from email_classifier.ingestion_service.parsing import parse_threads
from email_classifier.ingestion_service.summarizer import render_summary_md, render_user_day_summary_md
from email_classifier.shared.logging import setup_logging
from email_classifier.shared.config import warn_if_missing_llm_keys, CONFIDENCE_MIN, MAX_LLM_THREAD_CHARS

P_RANK = {"P0":0, "P1":1, "P2":2, "P3":3}
logger = logging.getLogger("email_classifier.ingest")

def best_priority(triage: ThreadTriage) -> str:
    if not triage.actions:
        return "P3"
    return min((a.priority for a in triage.actions), key=lambda p: P_RANK[p])

def stabilize_rules(triage: ThreadTriage, thread_text_lower: str) -> ThreadTriage:
    # Deterministic urgency stabilizer (very simple, interview-friendly)
    if triage.email_type != "ACTION_REQUIRED" or not triage.action_required:
        triage.actions = []
        if triage.email_type == "INFORMATIONAL_ARCHIVE":
            triage.archive_recommendation = "archive"
        return triage

    urgent_hits = any(k in thread_text_lower for k in ["order today", "asap", "urgent", "deadline", "chasing", "sla", "complaint"])
    appt_hits = any(k in thread_text_lower for k in ["appointment", "book", "confirm", "wednesday", "thursday", "friday", "monday", "tuesday", "feb", "mar"])

    for a in triage.actions:
        if urgent_hits:
            a.priority = "P0" if a.priority in ["P1","P2","P3"] else a.priority
            a.blocking = True
            a.due = a.due or "ASAP"
        if appt_hits:
            if a.priority in ["P2","P3"]:
                a.priority = "P1"
            a.due = a.due or "Within 48h"
    return triage

def extract_participants(thread: EmailThread, handler_domains: List[str]) -> dict:
    handler_name = None
    handler_email = None
    customer_name = None
    customer_email = None

    # walk messages newest->oldest to get latest known names
    for m in reversed(thread.messages):
        sender = m.sent_from or ""
        dom = sender_domain(sender)
        sig_name = extract_signature_name(m.body or "")
        sal_name = extract_salutation_name(m.body or "")

        if dom and dom.lower() in handler_domains:
            if not handler_email:
                handler_email = sender
            if not handler_name and sig_name:
                handler_name = sig_name
        else:
            if not customer_email:
                customer_email = sender
            if not customer_name and sig_name:
                customer_name = sig_name
            if not customer_name and sal_name:
                customer_name = sal_name

    if not customer_name:
        customer_name = name_from_email(customer_email)

    return {
        "handler_name": handler_name,
        "handler_email": handler_email,
        "customer_name": customer_name,
        "customer_email": customer_email,
    }

def extract_user_email(thread: EmailThread, handler_domains: List[str], default_user: str | None) -> Optional[str]:
    if default_user:
        return default_user
    # Prefer an internal recipient (mailbox owner) from sent_to/sent_cc
    for m in reversed(thread.messages):
        for addr_list in (m.sent_to or [], m.sent_cc or []):
            for addr in addr_list:
                dom = sender_domain(addr)
                if dom and dom.lower() in handler_domains:
                    return addr
    # Fallback: latest internal sender
    for m in reversed(thread.messages):
        dom = sender_domain(m.sent_from)
        if dom and dom.lower() in handler_domains:
            return m.sent_from
    return None

def reconcile_participants(triage: ThreadTriage, participants: dict, handler_domains: List[str]) -> ThreadTriage:
    def is_handler_email(addr: str | None) -> bool:
        if not addr:
            return False
        dom = sender_domain(addr)
        return dom.lower() in handler_domains if dom else False

    # If LLM picked emails but swapped roles, correct using handler_domains
    if triage.handler_email and not is_handler_email(triage.handler_email):
        if not triage.customer_email:
            triage.customer_email = triage.handler_email
        triage.handler_email = participants.get("handler_email")

    if triage.customer_email and is_handler_email(triage.customer_email):
        if not triage.handler_email:
            triage.handler_email = triage.customer_email
        triage.customer_email = participants.get("customer_email")

    # Fill missing from deterministic extraction
    if not triage.handler_name:
        triage.handler_name = participants.get("handler_name")
    if not triage.handler_email:
        triage.handler_email = participants.get("handler_email")
    if not triage.customer_name:
        triage.customer_name = participants.get("customer_name")
    if not triage.customer_email:
        triage.customer_email = participants.get("customer_email")

    # Infer names from emails if still missing
    if not triage.customer_name:
        triage.customer_name = name_from_email(triage.customer_email)
    if not triage.handler_name:
        triage.handler_name = name_from_email(triage.handler_email)

    return triage

def normalize_triage(triage: ThreadTriage, ref_hint: str | None) -> ThreadTriage:
    # Ensure consistency between email_type and action_required
    if triage.email_type != "ACTION_REQUIRED" or not triage.action_required:
        triage.action_required = False
        triage.actions = []
        if triage.email_type == "INFORMATIONAL_ARCHIVE":
            triage.archive_recommendation = "archive"

    if triage.action_required and not triage.actions:
        triage.action_required = False
        triage.missing_info = list(set(triage.missing_info + ["action_details"]))

    # Prefer deterministic claim reference extracted from text over LLM output
    if ref_hint:
        if (not triage.thread_ref) or (not CLAIM_REF_RE.match((triage.thread_ref or "").strip())):
            triage.thread_ref = ref_hint
    elif triage.thread_ref:
        m = CLAIM_REF_RE.match((triage.thread_ref or "").strip())
        triage.thread_ref = m.group(0).upper() if m else triage.thread_ref
    return triage


def format_thread_for_llm(thread: EmailThread, handler_domains: List[str], max_chars: int = MAX_LLM_THREAD_CHARS) -> tuple[str, str, str]:
    # returns (messages_text, thread_ref_hint, latest_message_text)
    thread_ref_hint = None
    msgs_out = []
    latest_msg = None

    thread_id = thread.thread_id or ""
    for i, m in enumerate(thread.messages):
        subject = m.subject or ""
        body = (m.body or "")
        body = redact_for_llm(body)  # PII redaction before LLM
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...[truncated]"
        frm = m.sent_from or ""
        dom = sender_domain(frm)
        msgs_out.append(
            f"[{i}] THREAD_ID: {thread_id}\nFROM: {frm} (domain={dom})\nSUBJECT: {subject}\nBODY:\n{body}\n"
        )

    for m in reversed(thread.messages):
        thread_ref_hint = extract_claim_ref(m.subject, m.body)
        if thread_ref_hint:
            break

    if thread.messages:
        lm = thread.messages[-1]
        latest_msg = f"FROM: {lm.sent_from}\nSUBJECT: {lm.subject}\nBODY:\n{redact_for_llm(lm.body or '')}"

    return "\n---\n".join(msgs_out), (thread_ref_hint or ""), (latest_msg or "")

def build_thread_full_text(thread: EmailThread) -> str:
    parts = []
    for i, m in enumerate(thread.messages):
        parts.append(
            "\n".join([
                f"[{i}] FROM: {m.sent_from or ''}",
                f"SUBJECT: {m.subject or ''}",
                f"SENT_AT: {m.sent_at or ''}",
                f"BODY:\n{redact_for_index(m.body or '')}",
            ])
        )
    return "\n---\n".join(parts)

def stable_thread_key(thread: EmailThread) -> str:
    if thread.thread_id:
        return thread.thread_id
    msg_ids = [m.message_id for m in thread.messages if m.message_id]
    if msg_ids:
        base = "|".join(sorted(msg_ids))
    else:
        parts = []
        for i, m in enumerate(thread.messages):
            parts.append(f"{i}|{m.sent_from or ''}|{m.subject or ''}|{m.sent_at or ''}|{(m.body or '')[:200]}")
        base = "|".join(parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def stable_uuid(thread_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"email_classifier:{thread_key}"))

def build_participants_text(triage: ThreadTriage, participants: dict, user_email: Optional[str]) -> str:
    def _keep(v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        return None if "@" in v else v

    vals = [
        _keep(triage.handler_name),
        _keep(triage.customer_name),
        _keep(triage.entities.get("counterparty") if isinstance(triage.entities, dict) else None),
        _keep(participants.get("handler_name")),
        _keep(participants.get("customer_name")),
    ]
    return " | ".join([v for v in vals if v])

def upsert_by_uuid(col, uuid_str: str, properties: dict) -> None:
    try:
        col.data.insert(uuid=uuid_str, properties=properties)
    except Exception as e:
        if "already exists" in str(e).lower():
            col.data.update(uuid=uuid_str, properties=properties)
        else:
            raise

def main():
    load_dotenv()
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest + triage emails.json -> Weaviate + summary outputs")
    parser.add_argument("--input", required=True, help="Path to email JSON")
    parser.add_argument("--outdir", default=os.getenv("OUT_DIR","out"))
    parser.add_argument("--no-weaviate", action="store_true", help="Skip vector DB indexing")
    args = parser.parse_args()

    warn_if_missing_llm_keys()
    model = os.getenv("LLM_MODEL","gpt-4o-mini")
    handler_domains = [d.strip().lower() for d in os.getenv("HANDLER_DOMAINS","").split(",") if d.strip()]

    ensure_dir(args.outdir)
    if not args.no_weaviate:
        from email_classifier.weaviate_service.weaviate_service import ensure_schema, THREAD_CLASS, DETAIL_CLASS
        from email_classifier.weaviate_service.weaviate_client import get_client
        ensure_schema()

    try:
        raw = load_json(args.input)
        threads = parse_threads(raw)
    except Exception as e:
        logger.exception("Failed to load/parse input JSON: %s", e)
        raise

    client = None
    threads_col = None
    detail_col = None
    summary_col = None
    if not args.no_weaviate:
        try:
            from email_classifier.weaviate_service.weaviate_service import THREAD_CLASS, DETAIL_CLASS, DAILY_SUMMARY_CLASS
            from email_classifier.weaviate_service.weaviate_client import get_client
            client = get_client()
            client.connect()
            threads_col = client.collections.get(THREAD_CLASS)
            detail_col = client.collections.get(DETAIL_CLASS)
            summary_col = client.collections.get(DAILY_SUMMARY_CLASS)
        except Exception as e:
            raise RuntimeError(f"Weaviate connection failed: {e}") from e

    actions_log = os.path.join(args.outdir, "actions.jsonl")
    if os.path.exists(actions_log):
        os.remove(actions_log)

    triages: List[ThreadTriage] = []

    default_user = os.getenv("MAILBOX_EMAIL") or None
    triage_index: List[dict] = []
    for idx, thread in enumerate(tqdm(threads, desc="Triaging threads")):
        try:
            messages_text, ref_hint, latest_msg = format_thread_for_llm(thread, handler_domains)
            participants = extract_participants(thread, handler_domains)
            user_email = extract_user_email(thread, handler_domains, default_user)

            user_prompt = THREAD_TRIAGE_USER.format(
                schema=schema_str(ThreadTriage),
                handler_domains=handler_domains,
                few_shot_examples=FEW_SHOT_EXAMPLES,
                messages=messages_text
            )

            raw_out = call_llm_json_model(
                model=model,
                system=THREAD_TRIAGE_SYSTEM,
                user=user_prompt,
                model_cls=ThreadTriage,
                temperature=0.1,
            )
            triage = raw_out
            if triage.confidence < CONFIDENCE_MIN:
                triage.needs_human_review = True
            if triage.missing_info:
                triage.needs_human_review = True

            triage = normalize_triage(triage, ref_hint)
            triage = reconcile_participants(triage, participants, handler_domains)

            # best-effort counterparty from latest external sender if missing
            if isinstance(triage.entities, dict) and not triage.entities.get("counterparty"):
                cp = None
                for m in reversed(thread.messages):
                    dom = sender_domain(m.sent_from)
                    if dom and dom.lower() not in handler_domains:
                        cp = m.sent_from
                        break
                if cp:
                    triage.entities["counterparty"] = cp

            triage = stabilize_rules(triage, messages_text.lower())
            triages.append(triage)

            thread_key = stable_thread_key(thread)
            thread_uuid = stable_uuid(thread_key)
            if threads_col is not None and detail_col is not None:
                # Upsert thread object (Weaviate handles vectorization)
                # Compute latest/earliest sent_at
                dts = [parse_datetime(m.sent_at) for m in thread.messages]
                dts = [d for d in dts if d is not None]
                latest_dt = max(dts) if dts else None
                earliest_dt = min(dts) if dts else None
                participants_text = build_participants_text(triage, participants, user_email)
                upsert_by_uuid(
                    threads_col,
                    thread_uuid,
                    {
                        "thread_id": thread.thread_id or "",
                        "thread_key": thread_key or "",
                        "thread_ref": triage.thread_ref or "",
                        "email_type": triage.email_type,
                        "topic": triage.topic or "",
                        "topic_lc": (triage.topic or "").lower(),
                        "category": triage.category or "",
                        "priority_best": best_priority(triage),
                        "action_required": triage.action_required,
                        "actions_json": json.dumps([a.model_dump() for a in triage.actions], ensure_ascii=False),
                        "actions_text": " | ".join([a.action_item for a in triage.actions if a.action_item]),
                        "counterparty": (triage.entities.get("counterparty") if isinstance(triage.entities, dict) else "") or "",
                        "counterparty_lc": ((triage.entities.get("counterparty") if isinstance(triage.entities, dict) else "") or "").lower(),
                        "entities_json": json.dumps(triage.entities, ensure_ascii=False),
                        "thread_summary": triage.rationale or "",
                        "latest_message": latest_msg,
                        "participants_text": participants_text,
                        "latest_sent_at": latest_dt.isoformat() if latest_dt else "",
                        "confidence": triage.confidence,
                        "archive_recommendation": triage.archive_recommendation,
                        "urgency_reason": triage.urgency_reason or "",
                        "missing_info_json": json.dumps(triage.missing_info, ensure_ascii=False),
                        "handler_name": triage.handler_name or "",
                        "handler_name_lc": (triage.handler_name or "").lower(),
                        "handler_email": triage.handler_email or "",
                        "customer_name": triage.customer_name or "",
                        "customer_name_lc": (triage.customer_name or "").lower(),
                        "customer_email": triage.customer_email or "",
                        "user_email": user_email or "",
                        "user_email_lc": (user_email or "").lower(),
                    },
                )

                upsert_by_uuid(
                    detail_col,
                    thread_uuid,
                    {
                        "thread_id": thread.thread_id or "",
                        "thread_key": thread_key or "",
                        "thread_ref": triage.thread_ref or "",
                        "user_email": user_email or "",
                        "user_email_lc": (user_email or "").lower(),
                        "full_text": build_thread_full_text(thread),
                        "participants_text": participants_text,
                        "messages_json": json.dumps([m.model_dump() for m in thread.messages], ensure_ascii=False),
                        "latest_sent_at": latest_dt.isoformat() if latest_dt else "",
                        "earliest_sent_at": earliest_dt.isoformat() if earliest_dt else "",
                        "message_count": len(thread.messages),
                    },
                )
            triage_index.append({
                "triage": triage,
                "user_email": user_email,
                "latest_dt": latest_dt,
            })

            append_jsonl(actions_log, {
                "thread_id": thread.thread_id,
                "thread_ref": triage.thread_ref,
                "email_type": triage.email_type,
                "topic": triage.topic,
                "action_required": triage.action_required,
                "actions": [a.model_dump() for a in triage.actions],
                "archive_recommendation": triage.archive_recommendation,
                "urgency_reason": triage.urgency_reason,
                "missing_info": triage.missing_info,
                "confidence": triage.confidence,
                "rationale": triage.rationale,
                "entities": triage.entities,
                "handler_name": triage.handler_name,
                "handler_email": triage.handler_email,
                "customer_name": triage.customer_name,
                "customer_email": triage.customer_email,
            })
        except Exception as e:
            logger.exception("Failed processing thread_id=%s: %s", getattr(thread, "thread_id", None), e)
            continue

    # Write summary artifacts
    out_threads = os.path.join(args.outdir, "threads_output.json")
    write_json(out_threads, [t.model_dump() for t in triages])

    summary_md = render_summary_md(triages)
    out_summary = os.path.join(args.outdir, "daily_summary.md")
    write_text(out_summary, summary_md)

    # User/day summaries (stored in Weaviate for retrieval)
    if summary_col is not None and triage_index:
        from datetime import datetime
        grouped: dict[tuple[str, str], List[ThreadTriage]] = {}
        for row in triage_index:
            triage = row["triage"]
            user_email = row["user_email"] or ""
            dt = row["latest_dt"]
            day = dt.date().isoformat() if dt else datetime.now().date().isoformat()
            key = (user_email, day)
            grouped.setdefault(key, []).append(triage)

        created_at = datetime.now().isoformat()
        for (user_email, day), group in grouped.items():
            md = render_user_day_summary_md(group, user_email or "(unknown)", day)
            actionable = [t for t in group if t.email_type=="ACTION_REQUIRED" and t.action_required]
            info = [t for t in group if t.email_type=="INFORMATIONAL_ARCHIVE"]
            irr = [t for t in group if t.email_type=="IRRELEVANT"]
            summary_col.data.insert(
                properties={
                    "day": day,
                    "user_email": user_email or "",
                    "user_email_lc": (user_email or "").lower(),
                    "summary_md": md,
                    "total_threads": len(group),
                    "action_required": len(actionable),
                    "informational": len(info),
                    "irrelevant": len(irr),
                    "created_at": created_at,
                }
            )

    if client:
        client.close()
    print(f"\nDone.\n- {out_summary}\n- {actions_log}\n- {out_threads}\nWeaviate populated at {os.getenv('WEAVIATE_URL')}")
if __name__ == "__main__":
    main()
