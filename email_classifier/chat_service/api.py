from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from email_classifier.langchain_agent.agent import LangChainAgent
from email_classifier.shared.config import warn_if_missing_llm_keys
from email_classifier.shared.logging import set_request_id, setup_logging
from email_classifier.weaviate_service.weaviate_service import ensure_schema

load_dotenv()
setup_logging()
warn_if_missing_llm_keys()
app = FastAPI(title="Email Ops Agent (LangChain)")
logger = logging.getLogger("email_classifier.api")

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    set_request_id(req_id)
    response = await call_next(request)
    response.headers["x-request-id"] = req_id
    return response

lc_agent = LangChainAgent(max_steps=10)

@app.on_event("startup")
async def startup_checks():
    try:
        ensure_schema()
    except Exception as e:
        # Keep API up so health/debug endpoints still work, but log clear startup issue.
        logger.exception("Startup schema ensure failed: %s", e)

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    history: list[dict[str, str]] | None = None

class AskResponse(BaseModel):
    answer: str

@app.post("/ask_langchain", response_model=AskResponse)
async def ask_langchain(req: AskRequest):
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        question = req.question
        ans = await lc_agent.arun(question, model, req.session_id)
        if req.session_id:
            _MEMORY.setdefault(req.session_id, []).append({"role": "assistant", "content": ans})
        return AskResponse(answer=ans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain agent error: {e}") from e

# In-memory chat history (per session_id)
_MEMORY: dict[str, list[dict[str, str]]] = {}

@app.get("/health")
def health():
    return {"status": "ok"}
