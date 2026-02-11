from __future__ import annotations
import os
import sys


def run_api() -> None:
    """Run FastAPI server via uvicorn."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    import uvicorn  # lazy import

    uvicorn.run("email_classifier.chat_service.api:app", host=host, port=port, reload=False)


def run_ingest() -> None:
    """Run ingestion entrypoint."""
    from email_classifier.ingestion_service.ingest import main

    main()


def main() -> None:
    """Dispatch CLI subcommands."""
    if len(sys.argv) < 2:
        print("Usage: email-classifier [api|ingest]")
        raise SystemExit(2)
    cmd = sys.argv[1].lower()
    if cmd == "api":
        run_api()
    elif cmd == "ingest":
        run_ingest()
    else:
        print(f"Unknown command: {cmd}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
