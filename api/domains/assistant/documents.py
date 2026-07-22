"""Persistence documents for synchronized Ask Rin-chan conversations."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import ClassVar, Literal

from beanie import Document
from pydantic import Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from api.shared.persistence.common import utc_now


class AskRinConversationDocument(Document):
    conversation_id: str
    user_id: str
    exercise_context_id: str
    summary: str = ""
    in_flight_request_id: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "ask_rin_conversations"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("conversation_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("exercise_context_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        ]


class AskRinMessageDocument(Document):
    message_id: str
    conversation_id: str
    user_id: str
    client_request_id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["complete", "interrupted"] = "complete"
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "ask_rin_messages"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("message_id", ASCENDING)], unique=True),
            IndexModel(
                [("user_id", ASCENDING), ("client_request_id", ASCENDING), ("role", ASCENDING)],
                unique=True,
            ),
            IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)]),
        ]


class AskRinRequestDocument(Document):
    client_request_id: str
    conversation_id: str
    user_id: str
    exercise_context_id: str
    status: Literal["in_progress", "completed", "interrupted", "refunded"]
    assistant_message_id: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "ask_rin_requests"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("user_id", ASCENDING), ("client_request_id", ASCENDING)], unique=True),
            IndexModel([("conversation_id", ASCENDING), ("created_at", DESCENDING)]),
        ]
