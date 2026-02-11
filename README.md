# Email Classifier

LLM-assisted triage for insurance operations email threads. It classifies emails, extracts actions and entities, prioritizes urgency, produces a handler-ready summary, and optionally supports free‑text Q&A over indexed threads.

## What It Produces
- `out/daily_summary.md` human‑readable workload summary
- `out/actions.jsonl` per‑thread actions and rationale
- `out/threads_output.json` full structured triage output
- Optional: Weaviate vector index for search/Q&A

## Requirements
- Python 3.10+
- An LLM API key compatible with LiteLLM
- Optional: Weaviate (via Docker) for retrieval/Q&A

## Install
```bash
pip install -r requirements.txt
```

## Environment
Set these in your shell or `.env`:
```bash
# LLM access (set one)
OPENAI_API_KEY=...
# or AZURE_OPENAI_API_KEY=...
# or ANTHROPIC_API_KEY=...
# or LITELLM_API_KEY=...

# Models
LLM_MODEL=gpt-4o-mini

# Handler domains (internal senders)
HANDLER_DOMAINS=aviva.com,internal.aviva.com

# Optional: Weaviate config
WEAVIATE_URL=http://localhost:8080
# Optional overrides
# WEAVIATE_HTTP_HOST=localhost
# WEAVIATE_HTTP_PORT=8080
# WEAVIATE_HTTP_SECURE=false
# WEAVIATE_GRPC_HOST=localhost
# WEAVIATE_GRPC_PORT=50051
# WEAVIATE_GRPC_SECURE=false

# Logging
LOG_LEVEL=INFO
```

## Run Ingestion (Generate Summary + Actions)
```bash
python ingestion_service/ingest.py --input /path/to/emails.json --outdir out
```

Or module form:
```bash
python -m ingestion_service.ingest --input /path/to/emails.json --outdir out
```

## Watch a Folder for New JSON Files
```bash
python ingestion_service/ingest.py --input /path/to/inbox --outdir out --watch --poll-interval 10
```

Each JSON file is processed into its own subfolder under `out/` (e.g., `out/emails_candidate/`).

If you want to run without Weaviate indexing:
```bash
python ingestion_service/ingest.py --input /path/to/emails.json --outdir out --no-weaviate
```

## Run Weaviate (Optional)
```bash
docker compose up -d
```

## Run Q&A API
```bash
uvicorn chat_service.api:app --host 0.0.0.0 --port 8000
```

Example:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Was there any action required for Broker X?"}'
```

## Run Streamlit UI
Start the API (above), then run:
```bash
streamlit run ui/streamlit_app.py
```

Optional: point the UI at a different API URL:
```bash
export AGENT_API_URL=http://localhost:8000
```

## Notes
- Classification and reasoning are handled by an LLM and validated against a strict JSON schema.
- PII redaction runs before sending content to the LLM.
- If Weaviate is disabled, outputs are still produced for the handler summary and actions.
