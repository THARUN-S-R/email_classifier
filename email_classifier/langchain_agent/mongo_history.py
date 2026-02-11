from __future__ import annotations
import os
from typing import List
from datetime import datetime

from pymongo import MongoClient
import atexit
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


_CLIENT: MongoClient | None = None

def _get_client(uri: str) -> MongoClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MongoClient(uri)
        atexit.register(_CLIENT.close)
    return _CLIENT

class MongoChatHistory(BaseChatMessageHistory):
    """Stores a full chat as a single MongoDB document per session_id."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.mongo_db = os.getenv("MONGO_DB", "email_classifier")
        self.mongo_coll = os.getenv("MONGO_COLLECTION", "chat_history")
        self._client = _get_client(self.mongo_uri)
        self._col = self._client[self.mongo_db][self.mongo_coll]

    @property
    def messages(self) -> List[BaseMessage]:
        doc = self._col.find_one({"session_id": self.session_id}) or {}
        msgs = doc.get("messages", [])
        out: List[BaseMessage] = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            if role == "human":
                out.append(HumanMessage(content=content))
            elif role == "ai":
                out.append(AIMessage(content=content))
            elif role == "system":
                out.append(SystemMessage(content=content))
        return out

    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "human"
        entry = {
            "role": role,
            "content": message.content,
            "ts": datetime.utcnow().isoformat(),
        }
        self._col.update_one(
            {"session_id": self.session_id},
            {
                "$setOnInsert": {"session_id": self.session_id},
                "$push": {"messages": entry},
            },
            upsert=True,
        )

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for m in messages:
            self.add_message(m)

    def clear(self) -> None:
        self._col.delete_one({"session_id": self.session_id})
