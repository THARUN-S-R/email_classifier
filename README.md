# Email Classifier

LLM-assisted triage for insurance operations email threads. It classifies emails, extracts actions and entities, prioritizes urgency, produces a handler-ready summary, and optionally supports free‑text Q&A over indexed threads.

## Project Description
This project implements an AI-driven Email Triage & Workload Management System designed to assist handlers in processing high-volume inboxes.

Rather than treating emails as simple text classification, the system models email handling as a multi-step cognitive workflow:

- Email Type Differentiation
- Action Extraction
- Topic & Intent Understanding
- Urgency & Importance Evaluation
- Priority Assignment
- Workload Aggregation
- Semantic Free-Text Question Answering

Built using:

- LangChain framework for agentic reasoning and tool usage
- Weaviate for hybrid semantic + structured retrieval
- LLMs for reasoning, interpretation, and summarisation

This project implements an AI-powered Email Triage & Workload Assistant that models email handling as a structured reasoning workflow rather than a simple classifier. The system differentiates email types, extracts required actions, interprets intent and topic, evaluates urgency/importance using hybrid AI-deterministic scoring, assigns priorities, aggregates handler workload, and enables semantic free-text querying over email threads via Weaviate hybrid search.

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
uv sync
```

## Build Package (uv)
The project is already configured as a package in `pyproject.toml` (Hatchling build backend).

Build source and wheel distributions:

```bash
uv build
```

Artifacts are written to `dist/`:

- `dist/email_classifier-<version>.tar.gz`
- `dist/email_classifier-<version>-py3-none-any.whl`

Optional local install from wheel:

```bash
uv pip install dist/*.whl
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
uv run email-classifier-ingest --input /path/to/emails.json --outdir out
```

Or module form:
```bash
uv run email-classifier-ingest --input /path/to/emails.json --outdir out
```

## Watch a Folder for New JSON Files
```bash
uv run email-classifier-ingest --input /path/to/inbox --outdir out --watch --poll-interval 10
```

Each JSON file is processed into its own subfolder under `out/` (e.g., `out/emails_candidate/`).



## Run Weaviate (Optional)
```bash
docker compose up -d
```

## Run Q&A API
```bash
uv run email-classifier-api
```

Example:
```bash
curl -X POST http://localhost:8000/ask_langchain \
  -H "Content-Type: application/json" \
  -d '{"question":"Was there any action required for Broker X?"}'
```

## Run Streamlit UI
Start the API (above), then run:
```bash
uv run streamlit run ui/streamlit_app.py
```

Optional: point the UI at a different API URL:
```bash
export AGENT_API_URL=http://localhost:8000
```

## Notes
- Classification and reasoning are handled by an LLM and validated against a strict JSON schema.
- PII redaction runs before sending content to the LLM.
- If Weaviate is disabled, outputs are still produced for the handler summary and actions.

## Developer Tooling
Formatting:
```bash
uv run black .
```

Linting:
```bash
uv run ruff check .
```

Type checking:
```bash
uv run mypy .
```
