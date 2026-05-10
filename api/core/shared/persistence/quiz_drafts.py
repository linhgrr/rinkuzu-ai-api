from __future__ import annotations

from typing import Any

from beanie.odm.enums import SortDirection
from loguru import logger

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


async def create_quiz_draft(doc: dict[str, Any]) -> dict[str, Any] | None:
    try:
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
    except Exception:
        logger.exception("[QuizDraftStore] create failed draft_id={}", doc.get("draft_id"))
        return None


async def load_quiz_draft_for_user(draft_id: str, user_id: str) -> dict[str, Any] | None:
    try:
        doc = await QuizDraftDocument.find_one(
            QuizDraftDocument.draft_id == draft_id,
            QuizDraftDocument.user_id == user_id,
        )
    except Exception:
        logger.exception(
            "[QuizDraftStore] load_for_user failed draft_id={} user_id={}", draft_id, user_id
        )
        return None
    return None if doc is None else _document_to_public_dict(doc)


async def update_quiz_draft_for_user(
    draft_id: str,
    user_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        doc = await QuizDraftDocument.find_one(
            QuizDraftDocument.draft_id == draft_id,
            QuizDraftDocument.user_id == user_id,
        )
        if doc is None:
            return None
        for key, value in updates.items():
            if key == "questions":
                setattr(doc, key, _normalize_questions(value))
            elif key == "progress":
                setattr(doc, key, QuizDraftProgressPayload.model_validate(value or {}))
            elif key == "pdf":
                setattr(doc, key, QuizDraftPdf.model_validate(value or {}))
            elif key == "status":
                setattr(doc, key, QuizDraftStatus(str(value)))
            else:
                setattr(doc, key, value)
        await doc.replace()
        return _document_to_public_dict(doc)
    except Exception:
        logger.exception(
            "[QuizDraftStore] update_for_user failed draft_id={} user_id={}", draft_id, user_id
        )
        return None


async def list_recent_quiz_drafts_for_user(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    try:
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
    except Exception:
        logger.exception("[QuizDraftStore] list_recent_for_user failed user_id={}", user_id)
        return []
    return [_document_to_public_dict(doc) for doc in docs]


async def delete_quiz_draft_for_user(draft_id: str, user_id: str) -> dict[str, Any] | None:
    try:
        doc = await QuizDraftDocument.find_one(
            QuizDraftDocument.draft_id == draft_id,
            QuizDraftDocument.user_id == user_id,
        )
        if doc is None:
            return None
        payload = _document_to_public_dict(doc)
        await doc.delete()
    except Exception:
        logger.exception(
            "[QuizDraftStore] delete_for_user failed draft_id={} user_id={}", draft_id, user_id
        )
        return None
    return payload
