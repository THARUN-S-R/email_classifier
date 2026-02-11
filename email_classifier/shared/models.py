from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Priority = Literal["P0", "P1", "P2", "P3"]
EmailType = Literal["ACTION_REQUIRED", "INFORMATIONAL_ARCHIVE", "IRRELEVANT"]

class EmailMessage(BaseModel):
    subject: str | None = None
    body: str | None = None
    sent_from: str | None = Field(default=None, alias="sent_from")
    sent_to: list[str] | None = None
    sent_cc: list[str] | None = None
    sent_at: str | None = None
    message_id: str | None = None
    attachments: list[dict[str, Any]] | None = None

    class Config:
        populate_by_name = True

class EmailThread(BaseModel):
    thread_id: str | None = None
    messages: list[EmailMessage]

class ActionItem(BaseModel):
    action_item: str
    owner: str = "handler"
    due: str | None = None
    priority: Priority
    blocking: bool = False

class ThreadTriage(BaseModel):
    thread_ref: str | None = None
    category: str | None = None
    email_type: EmailType
    topic: str | None = None

    action_required: bool = False
    actions: list[ActionItem] = Field(default_factory=list)

    entities: dict[str, Any] = Field(default_factory=dict)

    archive_recommendation: Literal["archive", "keep", "ignore"] = "keep"
    urgency_reason: str | None = None
    missing_info: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: str | None = None
    needs_human_review: bool = False

    handler_name: str | None = None
    handler_email: str | None = None
    customer_name: str | None = None
    customer_email: str | None = None

class QueryFilter(BaseModel):
    claim_ref: str | None = None
    counterparty: str | None = None
    handler_name: str | None = None
    handler_email: str | None = None
    customer_name: str | None = None
    customer_email: str | None = None
    user_email: str | None = None
    email_type: EmailType | None = None
    action_required: bool | None = None
    priority_at_most: Priority | None = None
    topic_contains: str | None = None
    sent_after: str | None = None
    sent_before: str | None = None
    sent_on: str | None = None

class ThreadSelection(BaseModel):
    thread_keys: list[str] = Field(default_factory=list)

class QueryPlan(BaseModel):
    thread_filter: dict[str, Any] | None = None
    detail_filter: dict[str, Any] | None = None
    summary_filter: dict[str, Any] | None = None
    search_query: str | None = None
    need_detail: bool | None = None
