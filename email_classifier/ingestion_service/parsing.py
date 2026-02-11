from __future__ import annotations
from typing import Any, List, Dict
from datetime import datetime, timezone
from email_classifier.shared.models import EmailThread, EmailMessage
from email_classifier.shared.utils import strip_html, parse_datetime

def parse_threads(raw: Any) -> List[EmailThread]:
    threads_raw: List[Any] = []

    if isinstance(raw, list):
        threads_raw = raw
    elif isinstance(raw, dict):
        if "threads" in raw and isinstance(raw["threads"], list):
            threads_raw = raw["threads"]
        elif "emails" in raw and isinstance(raw["emails"], list):
            threads_raw = raw["emails"]
        elif "messages" in raw and isinstance(raw["messages"], list):
            threads_raw = [raw]
        else:
            # numeric-key dict
            candidates = []
            for _, v in raw.items():
                if isinstance(v, dict) and isinstance(v.get("messages"), list):
                    candidates.append(v)
            if candidates:
                threads_raw = candidates
            else:
                raise ValueError("Unrecognized JSON structure.")
    else:
        raise ValueError("Input JSON must be list or dict.")

    out: List[EmailThread] = []
    for t in threads_raw:
        if not isinstance(t, dict) or not isinstance(t.get("messages"), list):
            continue
        msgs: List[EmailMessage] = []
        for m in t["messages"]:
            if not isinstance(m, dict):
                continue
            if "sent_at" not in m and "date_sent" in m:
                m["sent_at"] = m.get("date_sent")
            # Normalize recipients to lists
            for key in ("sent_to", "sent_cc"):
                if key in m:
                    val = m.get(key)
                    if isinstance(val, str):
                        m[key] = [v.strip() for v in val.split(",") if v.strip()]
                    elif isinstance(val, list):
                        m[key] = [str(v).strip() for v in val if str(v).strip()]
            body = m.get("body")
            if isinstance(body, str):
                m["body"] = strip_html(body)
            msgs.append(EmailMessage.model_validate(m))
        # Sort by sent_at when available; stable fallback to input order
        msgs_with_idx = []
        for idx, msg in enumerate(msgs):
            dt = parse_datetime(msg.sent_at)
            msgs_with_idx.append((idx, dt, msg))
        msgs_with_idx.sort(
            key=lambda im: (
                im[1] is None,
                im[1] or datetime.min.replace(tzinfo=timezone.utc),
                im[0],
            )
        )
        msgs = [m for _, _, m in msgs_with_idx]
        thread_id = t.get("thread_id") or t.get("id") or t.get("threadId")
        if not thread_id and msgs:
            thread_id = msgs[0].message_id or t.get("messages")[0].get("thread_id")
        out.append(EmailThread(thread_id=thread_id, messages=msgs))
    return out
