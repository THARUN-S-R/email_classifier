from __future__ import annotations
import logging, os
from pathlib import Path

def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.getenv("LOG_FILE", "app.log")

    root = logging.getLogger()
    if root.handlers:
        return

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    root.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)
