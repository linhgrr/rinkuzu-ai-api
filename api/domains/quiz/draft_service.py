"""Service for server-side quiz draft PDF processing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
import uuid

import fitz
from loguru import logger

from api.config import get_settings, normalize_endpoint
from api.exceptions import AppError
from api.shared import mongo_store
from api.shared.document_text import (
    ExtractedDocumentText,
    calculate_pdf_bytes_hash,
    extract_document_text_from_bytes_with_key_pool,
    load_or_extract_document_text_cached,
)
from api.shared.persistence import (
    create_quiz_draft,
    delete_quiz_draft_for_user,
    list_recent_quiz_drafts_for_user,
    load_quiz_draft_for_user,
    request_cancel_quiz_draft_for_user,
    update_quiz_draft_for_user,
)
from api.shared.persistence.common import is_storage_infra_error
from api.shared.s3 import get_quiz_draft_s3_client, get_s3_client

from .extraction import (
    build_extraction_prompt,
    invoke_document_text_extract_llm,
    validate_quiz_extract_dependencies,
)

if TYPE_CHECKING:
    from .schemas import QuizDraftCreateRequest, QuizDraftPatchRequest, QuizManualDraftCreateRequest

EXPIRY_HOURS = 48
TOTAL_STEPS = 3
SOURCE_STEP = 1
OCR_STEP = 2
SOURCE_DOWNLOAD_TIMEOUT_SEC = 180
SOURCE_ENDPOINT_TIMEOUT_SEC = 75
QUIZ_DRAFT_JOB_TIMEOUT_SEC = 600
_MAX_S3_KEY_LEN = 1024


class QuizDraftServiceError(Exception):
    """Base exception for draft service errors."""


class QuizDraftNotFoundError(QuizDraftServiceError):
    """Raised when a draft does not exist or is not owned by the user."""


class QuizDraftValidationError(QuizDraftServiceError):
    """Raised when user-supplied draft data is invalid."""


class QuizDraftConflictError(QuizDraftServiceError):
    """Raised when an autosave is based on a stale draft revision."""


class QuizDraftDependencyError(QuizDraftServiceError):
    """Raised when external services required by draft processing are unavailable."""


def _pdf_too_large_message(max_pdf_bytes: int) -> str:
    return f"Source PDF exceeds {max_pdf_bytes // (1024 * 1024)}MB limit."


def public_draft(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the public draft shape used by API responses."""
    pdf = doc.get("pdf") or {}
    return {
        "draft_id": doc.get("draft_id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "category_id": doc.get("category_id"),
        "prompt": doc.get("prompt"),
        "source_type": doc.get("source_type") or "pdf",
        "is_private": bool(doc.get("is_private", False)),
        "revision": int(doc.get("revision", 0)),
        "question_count": int(doc.get("question_count", len(doc.get("questions") or []))),
        "pdf": {
            "s3_key": pdf.get("s3_key"),
            "file_name": pdf.get("file_name"),
            "file_size": pdf.get("file_size"),
            "page_count": pdf.get("page_count"),
        },
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
        if not settings.object_storage_bucket:
            raise QuizDraftDependencyError("Object storage bucket is not configured.")

        max_pdf_bytes = int(settings.quiz_extract_max_pdf_bytes)
        max_pdf_mb = max_pdf_bytes // (1024 * 1024)

        normalized_key = self._normalize_and_validate_s3_key(req.s3_key, user_id)
        if req.file_size and req.file_size > max_pdf_bytes:
            raise QuizDraftValidationError(f"Invalid file size (max {max_pdf_mb}MB).")

        now = datetime.now(UTC)
        doc: dict[str, Any] = {
            "draft_id": uuid.uuid4().hex,
            "user_id": user_id,
            "title": req.title.strip(),
            "description": (req.description or "").strip(),
            "category_id": req.category_id,
            "prompt": (req.prompt or "").strip() or None,
            "source_type": "pdf",
            "is_private": getattr(req, "is_private", False),
            "revision": 0,
            "pdf": {
                "s3_key": normalized_key,
                "file_name": req.file_name.strip(),
                "file_size": req.file_size,
                "page_count": None,
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

        # Persistence propagates storage/programming errors; no false None on outage.
        return await create_quiz_draft(doc)

    async def create_manual_draft(
        self,
        req: QuizManualDraftCreateRequest,
        user_id: str,
    ) -> dict[str, Any]:
        """Create a durable private workspace without invoking extraction services."""
        if not mongo_store.is_available():
            raise QuizDraftDependencyError("MongoDB is not available.")

        now = datetime.now(UTC)
        return await create_quiz_draft(
            {
                "draft_id": uuid.uuid4().hex,
                "user_id": user_id,
                "title": req.title.strip(),
                "description": (req.description or "").strip(),
                "category_id": req.category_id,
                "prompt": None,
                "source_type": "manual",
                "is_private": req.is_private,
                "revision": 0,
                "pdf": {},
                "status": "drafting",
                "progress": {"processed": 0, "total": 0, "percent": 0},
                "questions": [],
                "error": None,
                "submitted_quiz_id": None,
                "created_at": now,
                "updated_at": now,
                "expires_at": None,
            }
        )

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
        if "description" in req.model_fields_set:
            updates["description"] = (req.description or "").strip()
        if "category_id" in req.model_fields_set:
            updates["category_id"] = req.category_id
        if req.questions is not None:
            updates["questions"] = req.questions
        if req.is_private is not None:
            updates["is_private"] = req.is_private

        if not updates:
            return existing

        updated = await update_quiz_draft_for_user(
            draft_id,
            user_id,
            updates,
            expected_revision=req.expected_revision,
        )
        if not updated:
            if req.expected_revision is not None:
                latest = await load_quiz_draft_for_user(draft_id, user_id)
                if latest is not None:
                    raise QuizDraftConflictError("Draft changed in another session.")
            raise QuizDraftNotFoundError("Draft not found.")
        return updated

    async def delete_draft(self, draft_id: str, user_id: str) -> dict[str, Any]:
        """Delete a draft: strict load → atomic cancel → local cancel → metadata.

        Observed delete (including concurrent convergence after observation) is
        200. Genuine absence is 404. Infrastructure/DB/task errors are 503.
        Blob cleanup is best-effort after metadata delete and never upgrades a
        successful delete into a false 500.
        """
        # Local import avoids circular import with draft_tasks → draft_service.
        from .draft_tasks import quiz_draft_task_manager

        try:
            draft = await load_quiz_draft_for_user(draft_id, user_id)
            if draft is None:
                raise QuizDraftNotFoundError("Draft not found.")

            cancelled = await request_cancel_quiz_draft_for_user(draft_id, user_id)
            if cancelled is not None:
                draft = cancelled
            # cancel None after observation: concurrent cleanup; keep observed draft.

            await quiz_draft_task_manager.cancel(draft_id)
            deleted = await delete_quiz_draft_for_user(draft_id, user_id)
        except QuizDraftNotFoundError:
            raise
        except AppError:
            raise
        except Exception as exc:
            if is_storage_infra_error(exc):
                logger.exception(
                    "[quiz_draft] delete_infra_failed draft_id={} user_id={}", draft_id, user_id
                )
                raise AppError(
                    code="service_unavailable",
                    message="Draft cleanup failed",
                    detail="Unable to complete draft delete; retry may succeed",
                    status_code=503,
                    meta={"retryable": True, "draft_id": draft_id},
                ) from exc
            # Programming / invariant errors propagate as generic 500.
            raise

        if deleted is None:
            # Metadata already gone after observation: concurrent convergence → 200.
            return draft

        pdf = deleted.get("pdf") or {}
        raw_s3_key = pdf.get("s3_key") if isinstance(pdf, dict) else None
        owned_key = self._owned_pdf_key_or_none(raw_s3_key, user_id)
        try:
            await asyncio.to_thread(self._delete_pdf_best_effort, owned_key)
        except Exception:
            logger.exception(
                "[quiz_draft] pdf_cleanup_schedule_failed draft_id={} user_id={}",
                draft_id,
                user_id,
            )
        return deleted

    @staticmethod
    def _owned_pdf_key_or_none(s3_key: str | None, user_id: str) -> str | None:
        """Return key only when it has the exact canonical owner prefix."""
        if not s3_key or not isinstance(s3_key, str):
            return None
        normalized = s3_key.strip()
        if not normalized or len(normalized) > _MAX_S3_KEY_LEN:
            return None
        prefix = f"uploads/quiz_extract/{user_id}/"
        if not normalized.startswith(prefix) or not normalized[len(prefix) :]:
            logger.warning(
                "[quiz_draft] skip_pdf_cleanup_non_owned_key user_id={}",
                user_id,
            )
            return None
        return normalized

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
            logger.info("[quiz_draft] processing_started draft_id={} user_id={}", draft_id, user_id)
            async with asyncio.timeout(QUIZ_DRAFT_JOB_TIMEOUT_SEC):
                await self._run_processing_stages(draft_id, user_id)
        except asyncio.CancelledError:
            await asyncio.shield(self._mark_interrupted(draft_id, user_id))
            logger.warning(
                "[quiz_draft] processing_interrupted draft_id={} user_id={}", draft_id, user_id
            )
            raise
        except TimeoutError:
            message = f"Quiz extraction exceeded {QUIZ_DRAFT_JOB_TIMEOUT_SEC} seconds."
            await self._mark_failed(draft_id, user_id, message)
            logger.error(
                "[quiz_draft] processing_timed_out draft_id={} user_id={}", draft_id, user_id
            )
        except Exception as exc:
            await self._mark_failed(draft_id, user_id, str(exc) or "Quiz extraction failed.")
            logger.exception(
                "[quiz_draft] processing_failed draft_id={} user_id={}", draft_id, user_id
            )

    async def _run_processing_stages(self, draft_id: str, user_id: str) -> None:
        draft = await self.get_draft(draft_id, user_id)
        if draft.get("status") in {"cancelled", "submitted", "completed"}:
            return

        await self._persist_or_raise(
            draft_id,
            user_id,
            {
                "status": "processing",
                "progress": {"processed": 0, "total": TOTAL_STEPS, "percent": 0},
                "error": None,
            },
        )

        settings = get_settings()
        s3_key = draft.get("pdf", {}).get("s3_key")
        if not s3_key:
            raise QuizDraftValidationError("PDF is missing.")
        bucket_name, model = self._require_processing_settings(settings)
        source_timeout = float(
            getattr(
                settings, "quiz_extract_source_download_timeout_sec", SOURCE_DOWNLOAD_TIMEOUT_SEC
            )
        )
        endpoint_timeout = min(
            source_timeout,
            float(
                getattr(
                    settings,
                    "quiz_extract_source_endpoint_timeout_sec",
                    SOURCE_ENDPOINT_TIMEOUT_SEC,
                )
            ),
        )
        logger.info(
            "[quiz_draft] source_download_started draft_id={} key={} timeout_sec={}",
            draft_id,
            s3_key,
            source_timeout,
        )
        try:
            async with asyncio.timeout(source_timeout):
                pdf_bytes, page_count = await self._load_source_pdf(
                    draft_id=draft_id,
                    settings=settings,
                    bucket_name=bucket_name,
                    s3_key=s3_key,
                    endpoint_timeout_sec=endpoint_timeout,
                )
        except TimeoutError as exc:
            raise QuizDraftDependencyError(
                f"Source download exceeded {source_timeout:g} seconds."
            ) from exc
        logger.info(
            "[quiz_draft] source_loaded draft_id={} bytes={} pages={}",
            draft_id,
            len(pdf_bytes),
            page_count,
        )
        await self._persist_or_raise(
            draft_id,
            user_id,
            {
                "pdf": {
                    **(draft.get("pdf") or {}),
                    "file_size": len(pdf_bytes),
                    "page_count": page_count,
                },
                "progress": {"processed": SOURCE_STEP, "total": TOTAL_STEPS, "percent": 33},
            },
        )

        filename = draft.get("pdf", {}).get("file_name") or "quiz-source.pdf"
        logger.info("[quiz_draft] ocr_started draft_id={} file={}", draft_id, filename)
        document_text = await self._load_or_extract_document_text(
            pdf_bytes=pdf_bytes,
            filename=filename,
        )
        logger.info(
            "[quiz_draft] ocr_completed draft_id={} chars={} cache_hit={}",
            draft_id,
            len(document_text.text),
            bool(document_text.metadata.get("ocr_cache_hit")),
        )
        await self._persist_or_raise(
            draft_id,
            user_id,
            {"progress": {"processed": OCR_STEP, "total": TOTAL_STEPS, "percent": 67}},
        )

        logger.info(
            "[quiz_draft] llm_started draft_id={} model={} chars={}",
            draft_id,
            model,
            len(document_text.text),
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

        await self._persist_or_raise(
            draft_id,
            user_id,
            {
                "status": "completed",
                "progress": {"processed": TOTAL_STEPS, "total": TOTAL_STEPS, "percent": 100},
                "questions": questions,
                "error": None,
            },
        )
        logger.info("[quiz_draft] processing_completed draft_id={} user_id={}", draft_id, user_id)

    @staticmethod
    async def _persist_or_raise(
        draft_id: str,
        user_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        updated = await update_quiz_draft_for_user(draft_id, user_id, updates)
        if not updated:
            raise QuizDraftDependencyError("Failed to persist quiz draft processing state.")
        return updated

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
            extract_document_text=lambda: extract_document_text_from_bytes_with_key_pool(
                pdf_bytes,
                filename=filename,
            ),
        )

    async def _mark_failed(self, draft_id: str, user_id: str, message: str) -> None:
        updated = await update_quiz_draft_for_user(
            draft_id,
            user_id,
            {
                "status": "failed",
                "progress": {"processed": 0, "total": TOTAL_STEPS, "percent": 0},
                "error": message,
            },
        )
        if not updated:
            logger.error(
                "[quiz_draft] failed_state_not_persisted draft_id={} user_id={}",
                draft_id,
                user_id,
            )

    async def _mark_interrupted(self, draft_id: str, user_id: str) -> None:
        latest = await load_quiz_draft_for_user(draft_id, user_id)
        if not latest or latest.get("status") in {"cancelled", "submitted", "completed"}:
            return
        updated = await update_quiz_draft_for_user(
            draft_id,
            user_id,
            {
                "status": "queued",
                "error": "Processing interrupted; queued for recovery.",
            },
        )
        if not updated:
            logger.error(
                "[quiz_draft] interrupted_state_not_persisted draft_id={} user_id={}",
                draft_id,
                user_id,
            )

    async def _load_source_pdf(
        self,
        *,
        draft_id: str,
        settings: Any,
        bucket_name: str,
        s3_key: str,
        endpoint_timeout_sec: float,
    ) -> tuple[bytes, int]:
        endpoints = self._source_endpoint_urls(settings)
        last_dependency_error: Exception | None = None
        for attempt, endpoint_url in enumerate(endpoints, start=1):
            logger.info(
                "[quiz_draft] source_endpoint_attempt draft_id={} attempt={}/{} endpoint={}",
                draft_id,
                attempt,
                len(endpoints),
                endpoint_url,
            )
            try:
                async with asyncio.timeout(endpoint_timeout_sec):
                    pdf_bytes = await asyncio.to_thread(
                        self._read_pdf_bytes,
                        get_quiz_draft_s3_client(endpoint_url=endpoint_url),
                        bucket_name,
                        s3_key,
                    )
            except TimeoutError as exc:
                last_dependency_error = exc
                logger.warning(
                    "[quiz_draft] source_endpoint_timed_out draft_id={} attempt={}/{} endpoint={} timeout_sec={}",
                    draft_id,
                    attempt,
                    len(endpoints),
                    endpoint_url,
                    endpoint_timeout_sec,
                )
            except QuizDraftDependencyError as exc:
                last_dependency_error = exc
                logger.warning(
                    "[quiz_draft] source_endpoint_failed draft_id={} attempt={}/{} endpoint={} error={}",
                    draft_id,
                    attempt,
                    len(endpoints),
                    endpoint_url,
                    str(exc),
                )
            else:
                page_count = await asyncio.to_thread(self._count_pdf_pages, pdf_bytes)
                return pdf_bytes, page_count

        raise QuizDraftDependencyError(
            "Unable to read PDF from object storage after trying all configured endpoints."
        ) from last_dependency_error

    @staticmethod
    def _source_endpoint_urls(settings: Any) -> list[str | None]:
        endpoints: list[str | None] = [
            getattr(settings, "object_storage_client_endpoint", None),
            getattr(settings, "object_storage_public_base_url", None),
            normalize_endpoint(
                getattr(settings, "object_storage_endpoint_internal", None),
                default_scheme="http",
            ),
        ]
        deduped: list[str | None] = []
        for endpoint in endpoints:
            if endpoint and endpoint not in deduped:
                deduped.append(endpoint)
        return deduped or [None]

    @staticmethod
    def _normalize_and_validate_s3_key(s3_key: str, user_id: str) -> str:
        normalized_key = s3_key.strip().lstrip("/")
        if not normalized_key or len(normalized_key) > _MAX_S3_KEY_LEN:
            raise QuizDraftValidationError("Forbidden s3_key for current user.")
        required_prefix = f"uploads/quiz_extract/{user_id}/"
        if (
            not normalized_key.startswith(required_prefix)
            or not normalized_key[len(required_prefix) :]
        ):
            raise QuizDraftValidationError("Forbidden s3_key for current user.")
        return normalized_key

    @staticmethod
    def _count_pdf_pages(pdf_bytes: bytes) -> int:
        try:
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
            raise QuizDraftDependencyError("Unable to read PDF from object storage.") from exc
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
