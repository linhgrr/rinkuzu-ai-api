"""
routers/pipeline.py — Content pipeline endpoints.
"""

import asyncio
from contextlib import suppress
from pathlib import Path
import re
import time
from typing import Annotated, Any
import uuid

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request
import fitz
import httpx
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from api.config import get_settings
from api.dependencies import (
    get_content_pipeline_availability,
    get_content_pipeline_service,
    get_current_user,
    get_session_manager,
    get_session_service,
)
from api.domains.content_pipeline.application.cancellation import JobCancelledError
from api.domains.content_pipeline.domain.errors import (
    PipelineSchedulingUnavailableError,
    PipelineSourceDownloadError,
    PipelineStaleWorkerError,
)
from api.domains.content_pipeline.domain.transitions import (
    CancelJobOutcome,
    RetryCompensationOutcome,
    RetryJobOutcome,
)
from api.exceptions import (
    AppError,
    PipelineNotCompletedError,
    PipelineNotFoundError,
    ServiceUnavailableError,
)
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import PathID
from api.shared import mongo_store
from api.shared.persistence import (
    compensate_failed_retry_reschedule,
    find_recent_active_job_by_source,
    load_pipeline_job_for_user,
    load_pipeline_job_status_for_user,
    load_subject_progress_for_user,
    request_cancel_pipeline_job_for_user,
    transition_pipeline_job_for_retry,
)
from api.shared.persistence.common import is_storage_infra_error
from api.shared.persistence.pipeline_jobs import list_recent_pipeline_jobs_all_status
from api.shared.url_fetch import UnsafeURLError, stream_download

from . import PipelineStatus
from .application.source_fetch import download_source_to_dir
from .schemas import (
    PipelineJobCancelResponse,
    PipelineJobListResponse,
    PipelineJobRetryResponse,
    PipelineJobStatusResponse,
    PipelineProcessResponse,
    PipelineRuntimeStatusResponse,
    PipelineSessionCreateResponse,
)

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_MAX_FILENAME_LENGTH = 200
_MAX_SUBJECT_ID_LENGTH = 120
_ASCII_CONTROL_MAX = 31
_ASCII_DELETE = 127
_SUBJECT_ID_PATTERN = re.compile(r"^[\w .,:;()\-\/+&']+$", re.UNICODE)

_LONG_POLL_STATUSES = {
    PipelineStatus.LOADING.value,
    PipelineStatus.CHUNKING.value,
    PipelineStatus.EXTRACTING.value,
    PipelineStatus.RANKING.value,
    PipelineStatus.VERIFYING.value,
    PipelineStatus.BUILDING_GRAPH.value,
    PipelineStatus.OPTIMIZING.value,
}

_ACTIVE_POLL_STATUSES = {
    PipelineStatus.EMBEDDING.value,
    PipelineStatus.MERGING.value,
}


def _validation_error(detail: str) -> AppError:
    return AppError(
        code="validation_error",
        message="Invalid request",
        detail=detail,
        status_code=400,
    )


def _pipeline_internal_error(detail: str) -> AppError:
    return AppError(
        code="internal_error",
        message="Internal server error",
        detail=detail,
        status_code=500,
    )


def _pipeline_storage_unavailable(
    detail: str, *, meta: dict[str, object] | None = None
) -> AppError:
    return AppError(
        code="service_unavailable",
        message="Storage unavailable",
        detail=detail,
        status_code=503,
        meta={"retryable": True, **(meta or {})},
    )


def _enforce_pdf_page_limit(file_path: Path, max_pages: int) -> int:
    """Validate the real PDF structure and page count, cleaning rejected uploads."""
    try:
        with fitz.open(file_path) as pdf_document:
            page_count = int(pdf_document.page_count)
    except Exception:
        with suppress(OSError):
            file_path.unlink()
        raise _validation_error("Uploaded file is not a readable PDF.") from None

    if page_count <= 0 or page_count > max_pages:
        with suppress(OSError):
            file_path.unlink()
        raise _validation_error(f"PDF must contain between 1 and {max_pages} pages.")
    return page_count


async def _load_owned_job_or_raise(job_id: str, user_id: str) -> dict[str, Any]:
    """Owner load: genuine miss → 404; classified storage failure → 503."""
    try:
        job_doc = await load_pipeline_job_for_user(job_id, user_id)
    except Exception as exc:
        if is_storage_infra_error(exc):
            logger.exception(
                "[PipelineRouter] load_for_user infra failed job_id={} user_id={}",
                job_id,
                user_id,
            )
            raise _pipeline_storage_unavailable(
                "Unable to load pipeline job; retry may succeed",
                meta={"job_id": job_id},
            ) from exc
        raise
    if not job_doc:
        raise PipelineNotFoundError(job_id)
    return job_doc


def _resolve_pipeline_retry_after_seconds(
    *,
    status_value: str,
    is_terminal: bool,
    is_delayed: bool,
) -> int:
    settings = get_settings()
    if is_terminal:
        return 0
    if is_delayed:
        return settings.content_pipeline_delayed_retry_after_sec
    if status_value in _LONG_POLL_STATUSES:
        return settings.content_pipeline_long_stage_retry_after_sec
    if status_value in _ACTIVE_POLL_STATUSES:
        return settings.content_pipeline_active_retry_after_sec
    return settings.content_pipeline_default_retry_after_sec


@router.get("/status", response_model=StandardResponse[PipelineRuntimeStatusResponse])
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def pipeline_status(
    request: Request,
    availability: Annotated[dict, Depends(get_content_pipeline_availability)],
) -> Any:
    """Check if content pipeline runtime modules are available."""
    del request
    available = availability["available"]
    return ok(
        {
            "available": available,
            "service_initialized": availability["service_initialized"],
        },
        meta={"message": "Content pipeline ready" if available else "Content pipeline unavailable"},
    )


class ProcessDocumentRequest(BaseModel):
    file_url: str
    filename: str
    subject_id: str | None = Field(default=None, max_length=_MAX_SUBJECT_ID_LENGTH)
    prs_threshold: float | None = None  # None = use checkpoint metadata threshold
    min_confidence: float = 0.6
    apply_reduction: bool = True
    page_batch_size: int = Field(default=10, ge=1, le=50)
    source_s3_key: str | None = None

    @field_validator("subject_id")
    @classmethod
    def normalize_subject_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > _MAX_SUBJECT_ID_LENGTH:
            raise ValueError(f"subject_id must be at most {_MAX_SUBJECT_ID_LENGTH} characters.")
        if any(
            ord(char) <= _ASCII_CONTROL_MAX or ord(char) == _ASCII_DELETE for char in normalized
        ):
            raise ValueError("subject_id cannot contain control characters.")
        if not _SUBJECT_ID_PATTERN.fullmatch(normalized):
            raise ValueError("subject_id contains unsupported characters.")
        return normalized


@router.post("/process", response_model=StandardResponse[PipelineProcessResponse], status_code=202)
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def process_document(  # noqa: C901
    request: Request,
    req: ProcessDocumentRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    availability: Annotated[dict, Depends(get_content_pipeline_availability)],
    pipeline_service: Any = Depends(get_content_pipeline_service),
    x_rinkuzu_pdf_page_limit: Annotated[int, Header(ge=1, le=100)] = 30,
) -> Any:
    """Run the full content processing pipeline from an S3 uploaded file."""
    _ = request  # SlowAPI requires the Request parameter for rate-limit context.
    if not availability["available"]:
        logger.error(
            "[PipelineRouter] Content pipeline unavailable",
            user_id=user_id,
            error=availability["error"],
            src=availability["src"],
            service_initialized=availability["service_initialized"],
        )
        raise AppError(
            code="service_unavailable",
            message="Service unavailable",
            detail="Content pipeline is unavailable.",
            status_code=503,
        )
    if not mongo_store.is_available():
        raise ServiceUnavailableError("MongoDB persistence")

    # Sanitize filename — strip directory components, enforce .pdf, reject NUL / path seps
    raw_name = Path(req.filename or "").name
    if not raw_name or not raw_name.lower().endswith(".pdf"):
        raise _validation_error("Only PDF files are supported.")
    if len(raw_name) > _MAX_FILENAME_LENGTH or any(c in raw_name for c in ("\x00", "/", "\\")):
        raise _validation_error("Invalid filename.")

    file_id = uuid.uuid4().hex[:8]
    save_path = (UPLOAD_DIR / f"{file_id}_{raw_name}").resolve()
    if not save_path.is_relative_to(UPLOAD_DIR.resolve()):
        raise _validation_error("Invalid filename.")

    # Safe download — validates scheme, blocks private IPs, enforces size cap
    settings = get_settings()
    try:
        await stream_download(
            req.file_url,
            save_path,
            max_bytes=settings.download_max_bytes,
            allowlist=settings.download_host_allowlist or None,
        )
    except UnsafeURLError as exc:
        logger.warning("[PipelineRouter] Rejected unsafe URL: {}", exc)
        raise _validation_error("URL not allowed.") from None
    except (httpx.HTTPError, OSError):
        logger.exception("[PipelineRouter] Failed to download file from {}", req.file_url)
        raise AppError(
            code="upstream_error",
            message="Upstream service error",
            detail="Failed to download file.",
            status_code=502,
        ) from None

    # Verify the downloaded file is actually a PDF (magic bytes check).
    try:
        async with aiofiles.open(save_path, "rb") as _f:
            _header = await _f.read(5)
        if not _header.startswith(b"%PDF-"):
            with suppress(OSError):
                save_path.unlink()
            raise _validation_error("Uploaded file is not a valid PDF.")
    except OSError:
        raise _pipeline_internal_error("Failed to verify uploaded file.") from None

    # The BFF derives this tier limit from the entitlement registry and sends it
    # over the service-token-authenticated channel. Count pages from the actual
    # downloaded file so a client cannot bypass the cap with forged metadata.
    effective_page_limit = min(
        x_rinkuzu_pdf_page_limit,
        settings.content_pipeline_max_pdf_pages,
    )
    _enforce_pdf_page_limit(save_path, effective_page_limit)

    async def _find_recent_duplicate(
        uid: str, source_s3_key: str, window_sec: int
    ) -> dict[str, Any] | None:
        return await find_recent_active_job_by_source(
            user_id=uid, source_s3_key=source_s3_key, window_sec=window_sec
        )

    try:
        job = await pipeline_service.start_job(
            file_path=str(save_path),
            subject_id=req.subject_id,
            prs_threshold=(
                req.prs_threshold if req.prs_threshold is not None else settings.prs_threshold
            ),
            min_confidence=req.min_confidence,
            apply_reduction=req.apply_reduction,
            user_id=user_id,
            content_processor_available=availability["available"],
            content_processor_src=availability["src"] or "",
            page_batch_size=req.page_batch_size,
            source_s3_key=req.source_s3_key,
            dedup_window_sec=settings.content_pipeline_dedup_window_sec,
            find_recent_duplicate=_find_recent_duplicate,
        )
    except (RuntimeError, ValueError, OSError):
        logger.exception("[PipelineRouter] Failed to initialize pipeline job for {}", req.filename)
        try:
            if save_path.exists():
                save_path.unlink()
        except OSError:
            logger.warning("[PipelineRouter] Failed to cleanup upload {}", save_path)
        raise _pipeline_internal_error("Failed to initialize pipeline job.") from None

    return ok(
        {
            "job_id": job.job_id,
            "filename": req.filename,
            "file_size": save_path.stat().st_size,
            "subject_id": job.subject_id,
            "status": job.status.value,
            "status_url": f"/api/v1/pipeline/jobs/{job.job_id}",
            "page_batch_size": job.page_batch_size,
            "retry_after_seconds": get_settings().content_pipeline_default_retry_after_sec,
        },
        meta={"message": "Processing started. Poll /api/v1/pipeline/jobs/{job_id} for progress."},
    )


@router.get("/jobs", response_model=StandardResponse[PipelineJobListResponse])
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def list_jobs(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Any:
    """List recent pipeline jobs of all statuses for the current user.

    Returns per-job live fields the frontend library needs: progress, retryable,
    eta_seconds, retry_after_seconds, is_terminal, is_delayed.
    """
    del request
    settings = get_settings()
    now = time.time()
    try:
        rows = await list_recent_pipeline_jobs_all_status(user_id=user_id, limit=limit)
    except Exception as exc:
        if is_storage_infra_error(exc):
            logger.exception(
                "[PipelineRouter] list_jobs infra failed user_id={}",
                user_id,
            )
            raise _pipeline_storage_unavailable(
                "Unable to list pipeline jobs; retry may succeed",
                meta={"user_id": user_id},
            ) from exc
        raise
    items = []
    for r in rows:
        status_value = r["status"]
        is_terminal = status_value in {"completed", "failed", "cancelled"}
        heartbeat = float(r.get("heartbeat_at") or r.get("updated_at") or now)
        is_delayed = (
            not is_terminal
            and heartbeat > 0
            and (now - heartbeat) >= settings.content_pipeline_job_delayed_after_sec
        )
        r["is_terminal"] = is_terminal
        r["is_delayed"] = is_delayed
        r["retry_after_seconds"] = _resolve_pipeline_retry_after_seconds(
            status_value=status_value, is_terminal=is_terminal, is_delayed=is_delayed
        )
        items.append(r)
    return ok({"jobs": items, "count": len(items)})


@router.get("/jobs/{job_id}", response_model=StandardResponse[PipelineJobStatusResponse])
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def get_job_status(
    request: Request,
    job_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
    *,
    include_debug: Annotated[bool, Query()] = False,
) -> Any:
    del request
    """Get pipeline job status and progress."""
    try:
        job_doc = await load_pipeline_job_status_for_user(
            job_id,
            user_id,
            include_debug=include_debug,
        )
    except Exception as exc:
        if is_storage_infra_error(exc):
            logger.exception(
                "[PipelineRouter] status load infra failed job_id={} user_id={}",
                job_id,
                user_id,
            )
            raise _pipeline_storage_unavailable(
                "Unable to load pipeline job status; retry may succeed",
                meta={"job_id": job_id},
            ) from exc
        raise
    if not job_doc:
        raise PipelineNotFoundError(job_id)

    settings = get_settings()
    now = time.time()
    status_value = job_doc.get("status", PipelineStatus.PENDING.value)
    created_at = float(job_doc.get("created_at") or 0.0)
    updated_at = float(job_doc.get("updated_at") or created_at or now)
    heartbeat_at = float(job_doc.get("heartbeat_at") or updated_at)
    is_terminal = status_value in {
        PipelineStatus.COMPLETED.value,
        PipelineStatus.FAILED.value,
        PipelineStatus.CANCELLED.value,
    }
    is_delayed = (
        not is_terminal
        and heartbeat_at > 0
        and (now - heartbeat_at) >= settings.content_pipeline_job_delayed_after_sec
    )
    retry_after_seconds = _resolve_pipeline_retry_after_seconds(
        status_value=status_value,
        is_terminal=is_terminal,
        is_delayed=is_delayed,
    )
    result = job_doc.get("result") or {}
    failed_batches = result.get("failed_batches")
    warnings = result.get("warnings")
    response = {
        "job_id": job_doc.get("job_id", job_id),
        "filename": job_doc.get("filename", ""),
        "subject_id": job_doc.get("subject_id", ""),
        "status": status_value,
        "current_step": job_doc.get("current_step", "Loaded from MongoDB"),
        "progress": job_doc.get("progress", 1.0 if status_value == "completed" else 0.0),
        "total_chunks": job_doc.get("total_chunks", 0),
        "page_batch_size": job_doc.get("page_batch_size", 10),
        "batch_count": job_doc.get("batch_count", 0),
        "failed_batch_count": job_doc.get("failed_batch_count", 0),
        "partial_success": bool(job_doc.get("partial_success", False)),
        "concepts_extracted": job_doc.get("concepts_extracted", 0),
        "concepts_after_merge": job_doc.get("concepts_after_merge", 0),
        "relations_verified": job_doc.get("relations_verified", 0),
        "graph_stats": job_doc.get("graph_stats", {}),
        "quality_report": job_doc.get("quality_report"),
        "debug_trace": job_doc.get("debug_trace") or [],
        "error_message": job_doc.get("error_message"),
        "error_code": job_doc.get("error_code"),
        "user_message": job_doc.get("user_message"),
        "eta_seconds": job_doc.get("eta_seconds"),
        "retry_count": job_doc.get("retry_count", 0),
        "retryable": bool(job_doc.get("retryable", False)),
        "failed_batches": failed_batches if isinstance(failed_batches, list) else [],
        "warnings": warnings if isinstance(warnings, list) else [],
        "is_terminal": is_terminal,
        "is_delayed": is_delayed,
        "created_at": created_at,
        "updated_at": updated_at,
        "heartbeat_at": heartbeat_at,
        "retry_after_seconds": retry_after_seconds,
        "partial_graph": job_doc.get("partial_graph"),
    }
    if status_value == PipelineStatus.COMPLETED.value and result:
        concept_map = result.get("concept_map", {})
        response["result"] = {
            "graph": result.get("graph", {"nodes": [], "edges": []}),
            "stats": result.get("stats", {}),
            "quality_report": result.get("quality_report"),
            "n_concepts": len(concept_map) if isinstance(concept_map, dict) else 0,
            "concepts_data": result.get("concepts_data", {}),
            "concept_map": concept_map,
            "concept_embedding_count": result.get("concept_embedding_count", 0),
            "prereq_edges": result.get("prereq_edges", []),
            "failed_batches": failed_batches if isinstance(failed_batches, list) else [],
            "warnings": warnings if isinstance(warnings, list) else [],
        }
    return ok(response)


@router.post(
    "/jobs/{job_id}/create-session", response_model=StandardResponse[PipelineSessionCreateResponse]
)
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def create_session_from_pipeline(
    request: Request,
    job_id: PathID,
    background_tasks: BackgroundTasks,
    max_steps: Annotated[int, Query(ge=5, le=10000)] = 9999,
    manager: Any = Depends(get_session_manager),
    exercise_svc: Any = Depends(get_session_service),
    user_id: str = Depends(get_current_user),
) -> Any:
    """Create a learning session from a completed pipeline job."""
    del request
    active_session = manager.get_active_pipeline_session(user_id, job_id)
    if active_session is not None:
        return ok(
            {
                "session_id": active_session.session_id,
                "n_concepts": len(active_session.concept_map),
                "source": "existing_session",
                "job_id": job_id,
                "status": "active",
            }
        )

    job_doc = await _load_owned_job_or_raise(job_id, user_id)

    if job_doc.get("status") != "completed":
        raise PipelineNotCompletedError(job_id, job_doc.get("status", "unknown"))

    result = job_doc.get("result")
    if not result:
        raise _pipeline_internal_error("Job found but has no result data.")

    for required_key in ("concepts_data", "concept_map", "prereq_edges"):
        if required_key not in result:
            logger.error(
                "[PipelineRouter] job_id={} missing_result_key={}",
                job_id,
                required_key,
            )
            raise _pipeline_internal_error("Pipeline result is incomplete.")

    concept_ids = set(result["concept_map"].keys())
    invalid_edges = [
        edge
        for edge in result["prereq_edges"]
        if edge.get("source") not in concept_ids or edge.get("target") not in concept_ids
    ]
    if invalid_edges:
        logger.warning(
            "[PipelineRouter] job_id={} invalid_edge_count={}",
            job_id,
            len(invalid_edges),
        )
        raise AppError(
            code="conflict",
            message="Conflict",
            detail="Pipeline graph validation failed. Please reprocess.",
            status_code=409,
        )

    try:
        subject_progress = await load_subject_progress_for_user(job_id, user_id)
    except Exception as exc:
        if is_storage_infra_error(exc):
            logger.exception(
                "[PipelineRouter] create-session progress load infra failed job_id={} user_id={}",
                job_id,
                user_id,
            )
            raise _pipeline_storage_unavailable(
                "Unable to load subject progress for session; retry may succeed",
                meta={"job_id": job_id},
            ) from exc
        raise
    session, created = await manager.get_or_create_pipeline_session(
        job_doc=job_doc,
        subject_progress=subject_progress,
        user_id=user_id,
        max_steps=max_steps,
    )

    if created:
        try:
            background_tasks.add_task(exercise_svc.eager_generate_first_exercise, session)
        except TypeError as exc:
            logger.warning("[PipelineRouter] Failed to schedule eager prefetch: {}", exc)

    return ok(
        {
            "session_id": session.session_id,
            "n_concepts": len(result["concept_map"]),
            "source": "new_session" if created else "existing_session",
            "job_id": job_id,
            "status": "active",
        }
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=StandardResponse[PipelineJobCancelResponse],
    status_code=202,
)
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def cancel_job(
    request: Request,
    job_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    """Request cancellation of a pipeline job.

    Works regardless of pipeline runtime availability so a stuck job can always
    be cancelled. Uses an owner-scoped atomic Mongo transition (not load-full-save).
    Terminal jobs are idempotent no-ops.
    """
    del request
    try:
        result = await request_cancel_pipeline_job_for_user(job_id, user_id)
    except Exception as exc:
        if is_storage_infra_error(exc):
            logger.exception(
                "[PipelineRouter] cancel infra failed job_id={} user_id={}", job_id, user_id
            )
            raise _pipeline_storage_unavailable(
                "Unable to cancel pipeline job; retry may succeed",
                meta={"job_id": job_id},
            ) from exc
        raise

    if result.outcome is CancelJobOutcome.NOT_FOUND:
        raise PipelineNotFoundError(job_id)
    if result.outcome is CancelJobOutcome.CONFLICT:
        raise AppError(
            code="conflict",
            message="Conflict",
            detail="Pipeline state changed while cancellation was requested. Please retry.",
            status_code=409,
            meta={"retryable": True, "job_id": job_id},
        )
    if result.outcome is CancelJobOutcome.ALREADY_TERMINAL:
        return ok(
            {"job_id": job_id, "status": result.status},
            meta={"message": "Job already terminal"},
        )
    return ok(
        {"job_id": job_id, "status": "cancelling"},
        meta={"message": "Cancellation requested"},
    )


def _raise_for_retry_transition_outcome(job_id: str, transition: Any) -> None:
    """Map pre-reschedule retry transition outcomes to HTTP errors."""
    if transition.outcome is RetryJobOutcome.NOT_FOUND:
        raise PipelineNotFoundError(job_id)
    if transition.outcome is RetryJobOutcome.INVALID_STATE:
        raise AppError(
            code="conflict",
            message="Conflict",
            detail="Job is not in a retryable state.",
            status_code=409,
        )
    if transition.outcome is RetryJobOutcome.NOT_RETRYABLE:
        raise _validation_error("This job cannot be retried.")
    if transition.outcome is RetryJobOutcome.MAX_RETRIES:
        raise _validation_error("Retry limit reached")
    if transition.outcome is RetryJobOutcome.NO_SOURCE:
        raise _validation_error("Job has no source to retry from")
    if transition.job is None:
        raise _pipeline_internal_error("Retry transition returned empty job payload")


async def _compensate_and_raise_retry_reschedule_failure(
    *,
    job_id: str,
    user_id: str,
    retry_count: int,
    exc: BaseException,
) -> None:
    """Safely compensate a retry generation, then classify the original error."""
    retryable = isinstance(
        exc,
        (
            PipelineSchedulingUnavailableError,
            PipelineSourceDownloadError,
            ConnectionError,
            OSError,
            TimeoutError,
        ),
    ) or is_storage_infra_error(exc)
    try:
        compensation = await compensate_failed_retry_reschedule(
            job_id,
            user_id,
            retry_count=retry_count,
            retryable=retryable,
        )
    except Exception as compensate_exc:
        logger.exception(
            "[PipelineRouter] retry compensation failed job_id={} user_id={} retry_count={}",
            job_id,
            user_id,
            retry_count,
        )
        if isinstance(exc, asyncio.CancelledError):
            raise exc from None
        if is_storage_infra_error(compensate_exc):
            raise _pipeline_storage_unavailable(
                "Unable to verify failed retry compensation; retry may succeed",
                meta={"job_id": job_id, "retry_count": retry_count},
            ) from compensate_exc
        raise

    logger.info(
        "[PipelineRouter] retry compensation job_id={} generation={} outcome={} "
        "persisted_status={} persisted_generation={} cancel_requested={}",
        job_id,
        retry_count,
        compensation.outcome.value,
        compensation.status,
        compensation.retry_count,
        compensation.cancel_requested,
    )

    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, JobCancelledError) or (
        isinstance(exc, PipelineStaleWorkerError)
        and compensation.outcome
        in {
            RetryCompensationOutcome.CANCEL_REQUESTED,
            RetryCompensationOutcome.STALE_GENERATION,
            RetryCompensationOutcome.WORKER_STARTED,
            RetryCompensationOutcome.ALREADY_TERMINAL,
        }
    ):
        raise AppError(
            code="conflict",
            message="Conflict",
            detail="Pipeline state changed while the retry was being scheduled.",
            status_code=409,
            meta={
                "retryable": True,
                "job_id": job_id,
                "state": compensation.outcome.value,
            },
        ) from exc

    if not retryable:
        # Programming/invariant failures remain 500s after compensation. Do not
        # relabel them as transient simply because they occurred during retry.
        raise exc

    logger.exception(
        "[PipelineRouter] retry reschedule failed job_id={} user_id={}",
        job_id,
        user_id,
    )
    raise _pipeline_storage_unavailable(
        "Unable to schedule retried pipeline job; retry may succeed",
        meta={
            "job_id": job_id,
            "retry_count": retry_count,
            "error_code": "pipeline_retry_reschedule_failed",
        },
    ) from exc


@router.post(
    "/jobs/{job_id}/retry",
    response_model=StandardResponse[PipelineJobRetryResponse],
    status_code=202,
)
@limiter.limit(get_settings().rate_limit_pipeline, exempt_when=is_admin_request)
async def retry_job_endpoint(
    request: Request,
    job_id: PathID,
    user_id: Annotated[str, Depends(get_current_user)],
    availability: Annotated[dict, Depends(get_content_pipeline_availability)],
    pipeline_service: Any = Depends(get_content_pipeline_service),
) -> Any:
    """Retry a terminal, retryable pipeline job by re-fetching its S3 source."""
    del request
    if not availability["available"]:
        raise ServiceUnavailableError("Content pipeline")
    settings = get_settings()
    try:
        transition = await transition_pipeline_job_for_retry(
            job_id,
            user_id,
            max_retry_count=settings.content_pipeline_max_retry_count,
        )
    except Exception as exc:
        if is_storage_infra_error(exc):
            logger.exception(
                "[PipelineRouter] retry infra failed job_id={} user_id={}", job_id, user_id
            )
            raise _pipeline_storage_unavailable(
                "Unable to retry pipeline job; retry may succeed",
                meta={"job_id": job_id},
            ) from exc
        raise

    _raise_for_retry_transition_outcome(job_id, transition)
    job = pipeline_service.build_job_from_payload(transition.job)
    try:
        await pipeline_service.reschedule_retried_job(
            job,
            download_source=download_source_to_dir,
        )
    except asyncio.CancelledError as exc:
        await _compensate_and_raise_retry_reschedule_failure(
            job_id=job_id,
            user_id=user_id,
            retry_count=int(job.retry_count),
            exc=exc,
        )
    except Exception as exc:
        await _compensate_and_raise_retry_reschedule_failure(
            job_id=job_id,
            user_id=user_id,
            retry_count=int(job.retry_count),
            exc=exc,
        )
    return ok(
        {
            "job_id": job_id,
            "status": job.status.value,
            "status_url": f"/api/v1/pipeline/jobs/{job_id}",
            "retry_count": job.retry_count,
        },
        meta={"message": "Retry started."},
    )
