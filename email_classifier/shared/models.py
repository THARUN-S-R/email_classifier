from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

Priority = Literal["P0", "P1", "P2", "P3"]
EmailType = Literal["ACTION_REQUIRED", "INFORMATIONAL_ARCHIVE", "IRRELEVANT"]

class EmailMessage(BaseModel):
    subject: Optional[str] = None
    body: Optional[str] = None
    sent_from: Optional[str] = Field(default=None, alias="sent_from")
    sent_to: Optional[List[str]] = None
    sent_cc: Optional[List[str]] = None
    sent_at: Optional[str] = None
    message_id: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None

    class Config:
        populate_by_name = True

class EmailThread(BaseModel):
    thread_id: Optional[str] = None
    messages: List[EmailMessage]

class ActionItem(BaseModel):
    action_item: str
    owner: str = "handler"
    due: Optional[str] = None
    priority: Priority
    blocking: bool = False

class ThreadTriage(BaseModel):
    thread_ref: Optional[str] = None
    category: Optional[str] = None
    email_type: EmailType
    topic: Optional[str] = None

    action_required: bool = False
    actions: List[ActionItem] = Field(default_factory=list)

    entities: Dict[str, Any] = Field(default_factory=dict)

    archive_recommendation: Literal["archive", "keep", "ignore"] = "keep"
    urgency_reason: Optional[str] = None
    missing_info: List[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: Optional[str] = None
    needs_human_review: bool = False

    handler_name: Optional[str] = None
    handler_email: Optional[str] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None

class QueryFilter(BaseModel):
    claim_ref: Optional[str] = None
    counterparty: Optional[str] = None
    handler_name: Optional[str] = None
    handler_email: Optional[str] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    user_email: Optional[str] = None
    email_type: Optional[EmailType] = None
    action_required: Optional[bool] = None
    priority_at_most: Optional[Priority] = None
    topic_contains: Optional[str] = None
    sent_after: Optional[str] = None
    sent_before: Optional[str] = None
    sent_on: Optional[str] = None

class ThreadSelection(BaseModel):
    thread_keys: List[str] = Field(default_factory=list)

class QueryPlan(BaseModel):
    thread_filter: Optional[Dict[str, Any]] = None
    detail_filter: Optional[Dict[str, Any]] = None
    summary_filter: Optional[Dict[str, Any]] = None
    search_query: Optional[str] = None
    need_detail: Optional[bool] = None
