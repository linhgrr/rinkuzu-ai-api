from __future__ import annotations

from typing import Any

from beanie import UpdateResponse
from beanie.odm.enums import SortDirection

from .common import normalize_for_bson, utc_now
from .documents import (
    QuizDraftDocument,
    QuizDraftListProjection,
    QuizDraftPdf,
    QuizDraftProgressPayload,
    QuizDraftStatus,
    QuizQuestion,
)


def _normalize_questions(
    questions: list[dict[str, Any]] | list[QuizQuestion] | None,
) -> list[QuizQuestion]:
    if not questions:
        return []
    normalized: list[QuizQuestion] = []
    for question in questions:
        if isinstance(question, QuizQuestion):
            normalized.append(question)
        else:
            normalized.append(QuizQuestion.model_validate(question))
    return normalized


def _normalize_update_payload(updates: dict[str, Any]) -> dict[str, Any]:
    """Validate nested fields and produce a BSON-safe ``$set`` payload."""
    payload: dict[str, Any] = {}
    for key, value in updates.items():
        if key == "questions":
            payload[key] = normalize_for_bson(_normalize_questions(value))
        elif key == "progress":
            payload[key] = normalize_for_bson(QuizDraftProgressPayload.model_validate(value or {}))
        elif key == "pdf":
            payload[key] = normalize_for_bson(QuizDraftPdf.model_validate(value or {}))
        elif key == "status":
            payload[key] = QuizDraftStatus(str(value)).value
        else:
            payload[key] = normalize_for_bson(value)
    return payload


def _document_to_public_dict(doc: QuizDraftDocument | QuizDraftListProjection) -> dict[str, Any]:
    questions = getattr(doc, "questions", [])
    return {
        "draft_id": doc.draft_id,
        "user_id": getattr(doc, "user_id", None),
        "title": doc.title,
        "description": doc.description,
        "category_id": doc.category_id,
        "prompt": doc.prompt,
        "pdf": doc.pdf.model_dump() if isinstance(doc.pdf, QuizDraftPdf) else dict(doc.pdf or {}),
        "status": doc.status.value if isinstance(doc.status, QuizDraftStatus) else str(doc.status),
        "progress": doc.progress.model_dump()
        if isinstance(doc.progress, QuizDraftProgressPayload)
        else dict(doc.progress or {}),
        "questions": [
            question.model_dump(by_alias=True)
            if isinstance(question, QuizQuestion)
            else dict(question)
            for question in questions
        ],
        "error": doc.error,
        "submitted_quiz_id": doc.submitted_quiz_id,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "expires_at": doc.expires_at,
    }


async def create_quiz_draft(doc: dict[str, Any]) -> dict[str, Any]:
    """Insert a quiz draft. Returns the public dict; DB/validation errors propagate.

    Never returns ``None`` for infrastructure failure — callers map infra to 503.
    """
    created = QuizDraftDocument(
        draft_id=str(doc["draft_id"]),
        user_id=str(doc["user_id"]),
        title=str(doc["title"]),
        description=str(doc.get("description") or ""),
        category_id=doc.get("category_id"),
        prompt=doc.get("prompt"),
        pdf=QuizDraftPdf.model_validate(doc.get("pdf") or {}),
        status=QuizDraftStatus(str(doc.get("status") or QuizDraftStatus.QUEUED.value)),
        progress=QuizDraftProgressPayload.model_validate(doc.get("progress") or {}),
        questions=_normalize_questions(doc.get("questions")),
        error=doc.get("error"),
        submitted_quiz_id=doc.get("submitted_quiz_id"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        expires_at=doc["expires_at"],
    )
    await created.insert()
    return _document_to_public_dict(created)


async def load_quiz_draft_for_user(draft_id: str, user_id: str) -> dict[str, Any] | None:
    """Strict owner-scoped lookup. ``None`` only for genuine absence; DB errors propagate."""
    doc = await QuizDraftDocument.find_one(
        QuizDraftDocument.draft_id == draft_id,
        QuizDraftDocument.user_id == user_id,
    )
    return None if doc is None else _document_to_public_dict(doc)


async def update_quiz_draft_for_user(
    draft_id: str,
    user_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """Atomically update only when owned and not already cancelled.

    The owner + ``status != cancelled`` predicate is evaluated with the ``$set``
    so a stale producer cannot resurrect a committed cancel. DB errors propagate.
    """
    set_payload = _normalize_update_payload(updates)
    set_payload.setdefault("updated_at", utc_now())
    doc = await QuizDraftDocument.find_one(
        QuizDraftDocument.draft_id == draft_id,
        QuizDraftDocument.user_id == user_id,
        {"status": {"$ne": QuizDraftStatus.CANCELLED.value}},
    ).update(
        {"$set": set_payload},
        response_type=UpdateResponse.NEW_DOCUMENT,
    )
    if doc is None:
        return None
    return _document_to_public_dict(doc)


async def request_cancel_quiz_draft_for_user(
    draft_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """Atomically set cancelled for one owned draft. DB errors propagate."""
    doc = await QuizDraftDocument.find_one(
        QuizDraftDocument.draft_id == draft_id,
        QuizDraftDocument.user_id == user_id,
    ).update(
        {
            "$set": {
                "status": QuizDraftStatus.CANCELLED.value,
                "updated_at": utc_now(),
            }
        },
        response_type=UpdateResponse.NEW_DOCUMENT,
    )
    if doc is None:
        return None
    return _document_to_public_dict(doc)


async def list_recent_quiz_drafts_for_user(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """List recent drafts. Empty list only for genuine absence; DB errors propagate."""
    docs = await (
        QuizDraftDocument.find(
            QuizDraftDocument.user_id == user_id,
            {
                "status": {
                    "$nin": [
                        QuizDraftStatus.EXPIRED.value,
                        QuizDraftStatus.SUBMITTED.value,
                        QuizDraftStatus.CANCELLED.value,
                    ]
                }
            },
            projection_model=QuizDraftListProjection,
        )
        .sort(("created_at", SortDirection.DESCENDING))
        .limit(limit)
        .to_list()
    )
    return [_document_to_public_dict(doc) for doc in docs]


async def list_recoverable_quiz_drafts(limit: int = 100) -> list[dict[str, Any]]:
    """Return queued/in-flight drafts that need an application-owned worker task."""
    docs = await (
        QuizDraftDocument.find(
            {
                "status": {
                    "$in": [
                        QuizDraftStatus.QUEUED.value,
                        QuizDraftStatus.PROCESSING.value,
                    ]
                }
            },
            projection_model=QuizDraftListProjection,
        )
        .sort(("created_at", SortDirection.ASCENDING))
        .limit(limit)
        .to_list()
    )
    return [_document_to_public_dict(doc) for doc in docs]


async def delete_quiz_draft_for_user(draft_id: str, user_id: str) -> dict[str, Any] | None:
    """Delete a draft owned by ``user_id``. ``None`` only when absent; DB errors propagate."""
    doc = await QuizDraftDocument.find_one(
        QuizDraftDocument.draft_id == draft_id,
        QuizDraftDocument.user_id == user_id,
    )
    if doc is None:
        return None
    payload = _document_to_public_dict(doc)
    await doc.delete()
    return payload
