from __future__ import annotations
import json, os, re
from typing import Any, Optional, Dict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup

CLAIM_REF_RE = re.compile(r"\bPIN-[A-Z]{3}-\d{5,}\b", re.IGNORECASE)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
SIGNOFF_RE = re.compile(r"^(regards|kind regards|thanks|thank you|best|sincerely|cheers|many thanks)[,!.]*$", re.IGNORECASE)
NAME_RE = re.compile(r"^[A-Z][a-z]+(?:\\s+[A-Z][a-z]+){1,3}$")
SALUTATION_RE = re.compile(r"^(dear|hi|hello)\\s+(mr|mrs|ms|dr)?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)", re.IGNORECASE)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text("\n").strip()
    return text.strip()

def extract_claim_ref(subject: Optional[str], body: Optional[str]) -> Optional[str]:
    for t in (subject or "", body or ""):
        m = CLAIM_REF_RE.search(t)
        if m:
            return m.group(0).upper()
    return None

def sender_domain(sender: Optional[str]) -> str:
    s = (sender or "").strip().lower()
    return s.split("@", 1)[1] if "@" in s else ""

def redact_pii(text: str) -> str:
    if not text:
        return ""
    t = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    t = PHONE_RE.sub("[REDACTED_PHONE]", t)
    return t

def redact_for_llm(text: str) -> str:
    return redact_pii(text)

def redact_for_index(text: str) -> str:
    return redact_pii(text)

def raw_store(text: str) -> str:
    return text or ""

def extract_signature_name(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    tail = lines[-10:]
    for line in reversed(tail):
        if SIGNOFF_RE.match(line):
            continue
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        if NAME_RE.match(line):
            return line
    return None

def extract_salutation_name(text: str) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines()[:3]:
        m = SALUTATION_RE.match(line.strip())
        if m:
            return m.group(3).strip()
    return None

def name_from_email(addr: Optional[str]) -> Optional[str]:
    if not addr or "@" not in addr:
        return None
    local = addr.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
    parts = [p for p in local.split() if p]
    if len(parts) >= 2:
        return " ".join(p.capitalize() for p in parts[:3])
    return None

def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try ISO first
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    # RFC2822 / email date formats
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # Common date/time formats
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None
