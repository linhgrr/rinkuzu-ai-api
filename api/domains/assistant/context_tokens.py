"""Confidential, user-bound exercise context tokens for Ask Rin-chan."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict, Field

from api.config import get_settings
from api.exceptions import AppError

_TOKEN_TTL_SECONDS = 24 * 60 * 60
_TOKEN_VERSION = 1


class ExerciseContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_id: str = Field(min_length=1, max_length=160)
    user_id: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=12_000)
    options: list[str] = Field(default_factory=list, max_length=20)
    concept_name: str | None = Field(default=None, max_length=500)
    bloom_level: int | None = Field(default=None, ge=1, le=6)
    question_image: str | None = Field(default=None, max_length=2_048)
    option_images: list[str | None] = Field(default_factory=list, max_length=20)
    session_id: str | None = Field(default=None, max_length=160)
    exercise_id: str | None = Field(default=None, max_length=160)
    rag_context: str = Field(default="", max_length=24_000)
    version: int = _TOKEN_VERSION


def _fernet() -> Fernet:
    secret = get_settings().internal_service_token
    if not secret:
        raise AppError(
            code="service_unavailable",
            message="Service unavailable",
            detail="Ask Rin-chan context encryption is not configured",
            status_code=503,
        )
    digest = hashlib.sha256(f"ask-rin-context-v1:{secret}".encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def issue_context_token(context: ExerciseContext) -> str:
    payload = context.model_dump_json(exclude_none=True).encode()
    return _fernet().encrypt(payload).decode()


def read_context_token(token: str, *, user_id: str) -> ExerciseContext:
    try:
        decrypted = _fernet().decrypt(token.encode(), ttl=_TOKEN_TTL_SECONDS)
        raw: Any = json.loads(decrypted)
        context = ExerciseContext.model_validate(raw)
    except (InvalidToken, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AppError(
            code="ask_rin_context_invalid",
            message="Exercise context is invalid or expired",
            detail="Refresh the exercise and try again",
            status_code=409,
        ) from exc

    if context.user_id != user_id:
        raise AppError(
            code="forbidden",
            message="Exercise context does not belong to this user",
            detail="The context token is bound to another account",
            status_code=403,
        )
    if context.version != _TOKEN_VERSION:
        raise AppError(
            code="ask_rin_context_stale",
            message="Exercise context is stale",
            detail="Refresh the exercise and try again",
            status_code=409,
        )
    return context


def quiz_context_id(*, question: str, options: list[str]) -> str:
    canonical = json.dumps(
        {"question": question.strip(), "options": [item.strip() for item in options]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"quiz:{hashlib.sha256(canonical.encode()).hexdigest()[:32]}"
