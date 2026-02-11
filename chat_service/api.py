from __future__ import annotations
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from chat_service.agent_graph import build_graph
from langchain_agent.agent import LangChainAgent
from shared.logging import setup_logging
from shared.config import warn_if_missing_llm_keys

load_dotenv()
setup_logging()
warn_if_missing_llm_keys()
app = FastAPI(title="Email Ops Agent (LangGraph)")

agent = build_graph()
lc_agent = LangChainAgent()

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        question = req.question
        out = agent.invoke({"question": question, "model": model})
        answer = out["answer"]
        if req.session_id:
            _MEMORY.setdefault(req.session_id, []).append({"role": "assistant", "content": answer})
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

@app.post("/ask_langchain", response_model=AskResponse)
def ask_langchain(req: AskRequest):
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        question = req.question
        ans = lc_agent.run(question, model, session_id=req.session_id)
        if req.session_id:
            _MEMORY.setdefault(req.session_id, []).append({"role": "assistant", "content": ans})
        return AskResponse(answer=ans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain agent error: {e}") from e

# In-memory chat history (per session_id)
_MEMORY: Dict[str, List[Dict[str, str]]] = {}

def _with_history(question: str, history: Optional[List[Dict[str, str]]], session_id: Optional[str]) -> str:
    if history is None and session_id:
        history = _MEMORY.get(session_id)
    if history:
        # Keep last 6 turns to limit prompt size
        h = history[-12:]
        parts = []
        for msg in h:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        parts.append(f"USER: {question}")
        combined = "\n".join(parts)
    else:
        combined = question
    if session_id:
        _MEMORY.setdefault(session_id, []).append({"role": "user", "content": question})
    return combined

@app.get("/health")
def health():
    return {"status": "ok"}
