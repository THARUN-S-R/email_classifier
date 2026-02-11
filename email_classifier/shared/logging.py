from __future__ import annotations
import logging
import os
import json
import contextvars
from datetime import datetime, timezone
from pathlib import Path

REQUEST_ID = contextvars.ContextVar("request_id", default=None)

def set_request_id(req_id: str) -> None:
    REQUEST_ID.set(req_id)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        req_id = REQUEST_ID.get()
        if req_id:
            payload["request_id"] = req_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt_kind = os.getenv("LOG_FORMAT", "json").lower()
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.getenv("LOG_FILE", "app.log")

    root = logging.getLogger()
    if root.handlers:
        return

    if fmt_kind == "plain":
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    else:
        fmt = JSONFormatter()
    root.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)
