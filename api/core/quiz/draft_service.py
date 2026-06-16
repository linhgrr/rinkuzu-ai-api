"""Service for server-side quiz draft PDF processing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
import uuid

import fitz
from loguru import logger

from api.config import get_settings
from api.core.quiz.extraction import (
    build_extraction_prompt,
    invoke_document_text_extract_llm,
    validate_quiz_extract_dependencies,
)
from api.core.shared import mongo_store
from api.core.shared.document_text import (
    ExtractedDocumentText,
    calculate_pdf_bytes_hash,
    extract_document_text_from_bytes,
    load_or_extract_document_text_cached,
)
from api.core.shared.persistence import (
    create_quiz_draft,
    delete_quiz_draft_for_user,
    list_recent_quiz_drafts_for_user,
    load_quiz_draft_for_user,
    update_quiz_draft_for_user,
)
from api.core.shared.s3 import get_s3_client

if TYPE_CHECKING:
    from api.schemas.quiz_draft import QuizDraftCreateRequest, QuizDraftPatchRequest

EXPIRY_HOURS = 48
TOTAL_STEPS = 1


class QuizDraftServiceError(Exception):
    """Base exception for draft service errors."""


class QuizDraftNotFoundError(QuizDraftServiceError):
    """Raised when a draft does not exist or is not owned by the user."""


class QuizDraftValidationError(QuizDraftServiceError):
    """Raised when user-supplied draft data is invalid."""


class QuizDraftDependencyError(QuizDraftServiceError):
    """Raised when external services required by draft processing are unavailable."""


def _pdf_too_large_message(max_pdf_bytes: int) -> str:
    return f"Source PDF exceeds {max_pdf_bytes // (1024 * 1024)}MB limit."


def public_draft(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the public draft shape used by API responses."""
    return {
        "draft_id": doc.get("draft_id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "category_id": doc.get("category_id"),
        "prompt": doc.get("prompt"),
        "pdf": doc.get("pdf") or {},
        "status": doc.get("status"),
        "progress": doc.get("progress") or {"processed": 0, "total": TOTAL_STEPS, "percent": 0},
        "questions": doc.get("questions") or [],
        "error": doc.get("error"),
        "submitted_quiz_id": doc.get("submitted_quiz_id"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "expires_at": doc.get("expires_at"),
    }


class QuizDraftService:
    """Owns the quiz draft lifecycle and background extraction state."""

    @staticmethod
    def _require_processing_settings(settings: Any) -> tuple[str, str]:
        bucket_name = settings.object_storage_bucket
        model = settings.llm_model
        if not bucket_name or not model:
            raise QuizDraftDependencyError("Quiz extraction dependencies are not configured.")
        return bucket_name, model

    async def create_draft(self, req: QuizDraftCreateRequest, user_id: str) -> dict[str, Any]:
        if not mongo_store.is_available():
            raise QuizDraftDependencyError("MongoDB is not available.")

        settings = get_settings()
        s3_client = get_s3_client()
        try:
            validate_quiz_extract_dependencies(settings, s3_client)
        except ValueError as exc:
            raise QuizDraftDependencyError(str(exc)) from exc
        bucket_name = settings.object_storage_bucket
        if not bucket_name:
            raise QuizDraftDependencyError("Object storage bucket is not configured.")

        max_pdf_bytes = int(settings.quiz_extract_max_pdf_bytes)
        max_pdf_mb = max_pdf_bytes // (1024 * 1024)

        normalized_key = self._normalize_and_validate_s3_key(req.s3_key, user_id)
        object_size = await asyncio.to_thread(
            self._head_pdf_object,
            s3_client,
            bucket_name,
            normalized_key,
        )
        if object_size <= 0:
            raise QuizDraftValidationError("Source PDF is empty.")
        if object_size > max_pdf_bytes:
            raise QuizDraftValidationError(_pdf_too_large_message(max_pdf_bytes))
        if req.file_size and req.file_size > max_pdf_bytes:
            raise QuizDraftValidationError(f"Invalid file size (max {max_pdf_mb}MB).")

        page_count = await asyncio.to_thread(
            self._count_pdf_pages,
            s3_client,
            bucket_name,
            normalized_key,
        )

        now = datetime.now(UTC)
        doc: dict[str, Any] = {
            "draft_id": uuid.uuid4().hex,
            "user_id": user_id,
            "title": req.title.strip(),
            "description": (req.description or "").strip(),
            "category_id": req.category_id,
            "prompt": (req.prompt or "").strip() or None,
            "pdf": {
                "s3_key": normalized_key,
                "file_name": req.file_name.strip(),
                "file_size": object_size,
                "page_count": page_count,
            },
            "status": "queued",
            "progress": {"processed": 0, "total": TOTAL_STEPS, "percent": 0},
            "questions": [],
            "error": None,
            "submitted_quiz_id": None,
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(hours=EXPIRY_HOURS),
        }

        created = await create_quiz_draft(doc)
        if not created:
            raise QuizDraftDependencyError("Failed to create draft.")
        return created

    async def get_draft(self, draft_id: str, user_id: str) -> dict[str, Any]:
        draft = await load_quiz_draft_for_user(draft_id, user_id)
        if not draft:
            raise QuizDraftNotFoundError("Draft not found.")
        return draft

    async def list_drafts(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return await list_recent_quiz_drafts_for_user(user_id, limit)

    async def patch_draft(
        self,
        draft_id: str,
        user_id: str,
        req: QuizDraftPatchRequest,
    ) -> dict[str, Any]:
        existing = await self.get_draft(draft_id, user_id)
        if existing.get("status") in {"cancelled", "submitted", "expired"}:
            raise QuizDraftValidationError("Draft can no longer be edited.")

        updates: dict[str, Any] = {}
        if req.title is not None:
            updates["title"] = req.title.strip()
        if req.description is not None:
            updates["description"] = req.description.strip()
        if req.category_id is not None:
            updates["category_id"] = req.category_id
        if req.questions is not None:
            updates["questions"] = req.questions

        if not updates:
            return existing

        updated = await update_quiz_draft_for_user(draft_id, user_id, updates)
        if not updated:
            raise QuizDraftNotFoundError("Draft not found.")
        return updated

    async def delete_draft(self, draft_id: str, user_id: str) -> dict[str, Any]:
        deleted = await delete_quiz_draft_for_user(draft_id, user_id)
        if not deleted:
            raise QuizDraftNotFoundError("Draft not found.")

        await asyncio.to_thread(self._delete_pdf_best_effort, deleted.get("pdf", {}).get("s3_key"))
        return deleted

    async def mark_submitted(self, draft_id: str, user_id: str, quiz_id: str) -> dict[str, Any]:
        existing = await self.get_draft(draft_id, user_id)
        # Idempotent: a draft that was already submitted keeps its original
        # quiz mapping. A retried submit must not overwrite submitted_quiz_id
        # nor re-delete the (already gone) source PDF — the caller reads the
        # returned submitted_quiz_id to avoid creating a duplicate quiz.
        if existing.get("status") == "submitted" and existing.get("submitted_quiz_id"):
            return existing
        updated = await update_quiz_draft_for_user(
            draft_id,
            user_id,
            {
                "status": "submitted",
                "submitted_quiz_id": quiz_id,
                "error": None,
            },
        )
        if not updated:
            raise QuizDraftNotFoundError("Draft not found.")
        await asyncio.to_thread(self._delete_pdf_best_effort, existing.get("pdf", {}).get("s3_key"))
        return updated

    async def process_draft(self, draft_id: str, user_id: str) -> None:
        """Run extraction in the background and persist final draft state."""
        try:
            draft = await self.get_draft(draft_id, user_id)
            if draft.get("status") in {"cancelled", "submitted", "completed"}:
                return

            await update_quiz_draft_for_user(
                draft_id,
                user_id,
                {"status": "processing", "error": None},
            )

            settings = get_settings()
            s3_key = draft.get("pdf", {}).get("s3_key")
            if not s3_key:
                raise QuizDraftValidationError("PDF is missing.")
            bucket_name, model = self._require_processing_settings(settings)
            pdf_bytes = await asyncio.to_thread(
                self._read_pdf_bytes,
                get_s3_client(),
                bucket_name,
                s3_key,
            )
            filename = draft.get("pdf", {}).get("file_name") or "quiz-source.pdf"
            document_text = await self._load_or_extract_document_text(
                pdf_bytes=pdf_bytes,
                filename=filename,
            )
            questions = await invoke_document_text_extract_llm(
                document_text=document_text,
                filename=filename,
                prompt=build_extraction_prompt(draft.get("prompt")),
                model=model,
            )
            if not questions:
                raise QuizDraftValidationError("No quiz questions extracted from PDF.")

            latest = await load_quiz_draft_for_user(draft_id, user_id)
            if not latest or latest.get("status") in {"cancelled", "submitted"}:
                return

            await update_quiz_draft_for_user(
                draft_id,
                user_id,
                {
                    "status": "completed",
                    "progress": {"processed": TOTAL_STEPS, "total": TOTAL_STEPS, "percent": 100},
                    "questions": questions,
                    "error": None,
                },
            )
            logger.info(
                "[quiz_draft] processing_completed draft_id={} user_id={}", draft_id, user_id
            )
        except Exception as exc:
            await self._mark_failed(draft_id, user_id, str(exc) or "Quiz extraction failed.")
            logger.exception(
                "[quiz_draft] processing_failed draft_id={} user_id={}", draft_id, user_id
            )

    async def _load_or_extract_document_text(
        self,
        *,
        pdf_bytes: bytes,
        filename: str,
    ) -> ExtractedDocumentText:
        return await load_or_extract_document_text_cached(
            file_hash=calculate_pdf_bytes_hash(pdf_bytes),
            file_name=filename,
            file_size_bytes=len(pdf_bytes),
            extract_document_text=lambda: asyncio.to_thread(
                extract_document_text_from_bytes,
                pdf_bytes,
                filename=filename,
            ),
        )

    async def _mark_failed(self, draft_id: str, user_id: str, message: str) -> None:
        await update_quiz_draft_for_user(
            draft_id,
            user_id,
            {
                "status": "failed",
                "progress": {"processed": 0, "total": TOTAL_STEPS, "percent": 0},
                "error": message,
            },
        )

    @staticmethod
    def _normalize_and_validate_s3_key(s3_key: str, user_id: str) -> str:
        normalized_key = s3_key.strip().lstrip("/")
        required_prefix = f"uploads/quiz_extract/{user_id}/"
        if not normalized_key.startswith(required_prefix):
            raise QuizDraftValidationError("Forbidden s3_key for current user.")
        return normalized_key

    @staticmethod
    def _head_pdf_object(s3_client: Any, bucket_name: str, s3_key: str) -> int:
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        except Exception as exc:
            raise QuizDraftValidationError("Unable to inspect PDF from s3_key.") from exc
        return int(response.get("ContentLength", 0))

    @staticmethod
    def _count_pdf_pages(s3_client: Any, bucket_name: str, s3_key: str) -> int:
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            body = response.get("Body")
            pdf_bytes = body.read() if body else b""
            if not pdf_bytes:
                raise ValueError("empty PDF body")
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                page_count = int(doc.page_count)
        except Exception as exc:
            raise QuizDraftValidationError("Unable to read PDF pages on server.") from exc
        return max(1, page_count)

    @staticmethod
    def _read_pdf_bytes(s3_client: Any, bucket_name: str, s3_key: str) -> bytes:
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            body = response.get("Body")
            pdf_bytes = body.read() if body else b""
        except Exception as exc:
            raise QuizDraftValidationError("Unable to read PDF for extraction.") from exc
        if not pdf_bytes:
            raise QuizDraftValidationError("PDF is empty.")
        max_pdf_bytes = int(get_settings().quiz_extract_max_pdf_bytes)
        if len(pdf_bytes) > max_pdf_bytes:
            raise QuizDraftValidationError(_pdf_too_large_message(max_pdf_bytes))
        return pdf_bytes

    @staticmethod
    def _delete_pdf_best_effort(s3_key: str | None) -> None:
        if not s3_key:
            return
        settings = get_settings()
        s3_client = get_s3_client()
        if not s3_client or not settings.object_storage_bucket:
            return
        try:
            s3_client.delete_object(Bucket=settings.object_storage_bucket, Key=s3_key)
            logger.info("[quiz_draft] deleted_pdf key={}", s3_key)
        except Exception:
            logger.exception("[quiz_draft] failed_to_delete_pdf key={}", s3_key)
