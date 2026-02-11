from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Email Ops Assistant", layout="wide")

st.title("📩 Email Ops Assistant (Triage + Agent Q&A)")
st.caption("Read-only assistant for prioritising workload and answering ad-hoc questions (grounded on ingested email threads).")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Agent API URL", value=DEFAULT_API_URL)
    timeout_s = st.slider("Request timeout (seconds)", 10, 180, 60)

    st.divider()
    st.header("Suggested questions")
    st.write("- Was there any action required for Border Loss Adjusters?")
    st.write("- Show actions for claim PIN-HOM-483912")
    st.write("- Why is PIN-MTR-552301 urgent? Show latest messages.")
    st.write("- List all P0 items")
    st.write("- What can I archive today?")

# Session state (NO type annotations on assignment)
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # list of {"role": "user"/"assistant", "content": str}

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

col_chat, col_debug = st.columns([2, 1], gap="large")

def call_agent(question: str) -> Dict[str, Any]:
    url = api_url.rstrip("/") + "/ask_langchain" #ask_langchain ask
    payload = {
        "question": question,
        "session_id": st.session_state["session_id"],
        "history": st.session_state["chat"],
    }
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()

with col_chat:
    st.subheader("Chat")

    chat_history: List[Dict[str, str]] = st.session_state["chat"]

    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if "last_latency_s" in st.session_state and st.session_state["last_latency_s"] is not None:
        st.caption(f"Last response time: {st.session_state['last_latency_s']:.2f}s")

    # Input handled at page bottom to keep it fixed in layout

with col_debug:
    st.subheader("Debug / Payload (optional)")
    st.caption("Useful for interview demo: show raw API response when needed.")

    if st.session_state["last_result"] is None:
        st.info("Ask a question to see the latest API response here.")
    else:
        st.json(st.session_state["last_result"])

    st.divider()
    st.subheader("Quick actions")
    if st.button("Clear chat"):
        st.session_state["chat"] = []
        st.session_state["last_result"] = None
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()

# Fixed input at page bottom
user_q = st.chat_input("Ask about workload, claims, brokers, actions, priorities…")
if user_q:
    chat_history.append({"role": "user", "content": user_q})
    with col_chat:
        with st.spinner("Thinking..."):
            try:
                t0 = time.time()
                data = call_agent(user_q)
                dt = time.time() - t0

                answer = (data.get("answer") or "").strip() or "No answer returned."
                chat_history.append({"role": "assistant", "content": answer})
                st.session_state["last_result"] = data
                st.session_state["last_latency_s"] = dt
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Agent API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
