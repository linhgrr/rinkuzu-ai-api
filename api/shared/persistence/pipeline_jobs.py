from __future__ import annotations

from datetime import timedelta
from typing import Any

from beanie import UpdateResponse
from beanie.odm.enums import SortDirection
from pymongo.errors import DuplicateKeyError

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.domains.content_pipeline.domain.transitions import (
    CancelJobOutcome,
    CancelJobResult,
    CreateJobOutcome,
    RetryCompensationOutcome,
    RetryCompensationResult,
    RetryJobOutcome,
    RetryJobResult,
    SaveJobOutcome,
)
from api.shared import mongo_store

from .common import (
    epoch_to_utc,
    normalize_for_bson,
    optional_epoch_to_utc,
    utc_now,
    utc_to_epoch,
)
from .document_chunks import delete_chunks_for_job
from .documents import (
    PipelineJobActiveProjection,
    PipelineJobCancelProjection,
    PipelineJobDocument,
    PipelineJobListProjection,
    PipelineJobLookupProjection,
    PipelineJobStatusProjection,
)
from .subject_progress import delete_subject_progress_for_user

_NON_TERMINAL_STATUSES = [s.value for s in PipelineStatus if not s.is_terminal]
_TERMINAL_STATUSES = [s.value for s in PipelineStatus if s.is_terminal]
_RETRYABLE_STATUSES = [PipelineStatus.FAILED.value, PipelineStatus.CANCELLED.value]


_RETRY_RESCHEDULE_FAILED_CODE = "pipeline_retry_reschedule_failed"
_RETRY_RESCHEDULE_FAILED_USER_MESSAGE = (
    "Retry was accepted but could not be scheduled. You can retry again."
)
_RETRY_RESCHEDULE_FAILED_ERROR = "Failed to download source or schedule retried pipeline job."


def _pipeline_job_update_ops(payload: dict[str, Any]) -> dict[str, Any]:
    """Build an atomic update that keeps ``cancel_requested`` monotonic.

    Other fields are replaced via ``$set``. ``cancel_requested`` uses ``$max`` so
    BSON false < true: false to true is allowed, true to false is impossible under
    concurrent interleaving. ``cancel_requested`` must not appear in ``$set``.

    State machine (worker saves):
    - Non-terminal progress may update only documents that are still non-terminal
      and share the same ``retry_count`` generation.
    - Terminal writes may transition only from non-terminal (same generation).
    - Authorized retry is the only path that clears ``cancel_requested`` (explicit
      ``$set`` outside this helper).
    """
    set_payload = dict(payload)
    cancel_requested = bool(set_payload.pop("cancel_requested", False))
    # Creation identity and source ownership are insert-only. A worker may only
    # update mutable lifecycle/result fields for its exact generation.
    for immutable_field in ("job_id", "user_id", "source_s3_key", "created_at"):
        set_payload.pop(immutable_field, None)
    return {
        "$set": set_payload,
        "$max": {"cancel_requested": cancel_requested},
    }


def pipeline_job_to_document(job: PipelineJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "subject_id": job.subject_id,
        "user_id": job.user_id,
        "status": job.status.value,
        "current_step": job.current_step,
        "progress": job.progress,
        "total_chunks": job.total_chunks,
        "total_pages": job.total_pages,
        "page_batch_size": job.page_batch_size,
        "batch_count": job.batch_count,
        "failed_batch_count": job.failed_batch_count,
        "partial_success": job.partial_success,
        "concepts_extracted": job.concepts_extracted,
        "concepts_after_merge": job.concepts_after_merge,
        "relations_verified": job.relations_verified,
        "graph_stats": normalize_for_bson(job.graph_stats)
        if isinstance(job.graph_stats, dict)
        else {},
        "quality_report": normalize_for_bson(job.quality_report),
        "debug_trace": normalize_for_bson(job.debug_trace),
        "result": normalize_for_bson(job.result),
        "partial_graph": normalize_for_bson(job.partial_graph),
        "error_message": job.error_message,
        "error_code": job.error_code,
        "user_message": job.user_message,
        "retryable": job.retryable,
        "created_at": epoch_to_utc(job.created_at),
        "updated_at": epoch_to_utc(job.updated_at),
        "heartbeat_at": epoch_to_utc(job.heartbeat_at),
        "completed_at": optional_epoch_to_utc(job.completed_at if job.status.is_terminal else None),
        "source_s3_key": job.source_s3_key,
        "prs_threshold": job.prs_threshold,
        "min_confidence": job.min_confidence,
        "apply_reduction": job.apply_reduction,
        "retry_count": job.retry_count,
        "cancel_requested": job.cancel_requested,
        "eta_seconds": job.eta_seconds,
    }


def _status_value(status: PipelineStatus | str) -> str:
    return status.value if isinstance(status, PipelineStatus) else str(status)


def _document_to_runtime_payload(doc: PipelineJobDocument) -> dict[str, Any]:
    return {
        "job_id": doc.job_id,
        "filename": doc.filename,
        "subject_id": doc.subject_id,
        "user_id": doc.user_id,
        "status": _status_value(doc.status),
        "current_step": doc.current_step,
        "progress": doc.progress,
        "total_chunks": doc.total_chunks,
        "total_pages": doc.total_pages,
        "page_batch_size": doc.page_batch_size,
        "batch_count": doc.batch_count,
        "failed_batch_count": doc.failed_batch_count,
        "partial_success": doc.partial_success,
        "concepts_extracted": doc.concepts_extracted,
        "concepts_after_merge": doc.concepts_after_merge,
        "relations_verified": doc.relations_verified,
        "graph_stats": normalize_for_bson(doc.graph_stats)
        if isinstance(doc.graph_stats, dict)
        else {},
        "quality_report": normalize_for_bson(doc.quality_report),
        "debug_trace": normalize_for_bson(doc.debug_trace),
        "result": normalize_for_bson(doc.result),
        "partial_graph": normalize_for_bson(doc.partial_graph),
        "error_message": doc.error_message,
        "error_code": doc.error_code,
        "user_message": doc.user_message,
        "retryable": doc.retryable,
        "created_at": utc_to_epoch(doc.created_at),
        "updated_at": utc_to_epoch(doc.updated_at),
        "heartbeat_at": utc_to_epoch(doc.heartbeat_at),
        "completed_at": utc_to_epoch(doc.completed_at, default=0.0) if doc.completed_at else None,
        "source_s3_key": doc.source_s3_key,
        "prs_threshold": doc.prs_threshold,
        "min_confidence": doc.min_confidence,
        "apply_reduction": doc.apply_reduction,
        "retry_count": doc.retry_count,
        "cancel_requested": doc.cancel_requested,
        "eta_seconds": doc.eta_seconds,
    }


def _status_row_to_runtime_payload(
    row: PipelineJobStatusProjection | dict[str, Any],
) -> dict[str, Any]:
    payload = (
        row.model_dump(mode="python") if isinstance(row, PipelineJobStatusProjection) else dict(row)
    )
    payload.pop("_id", None)

    status = payload.get("status")
    if isinstance(status, PipelineStatus):
        payload["status"] = status.value

    for field in ("created_at", "updated_at", "heartbeat_at"):
        value = payload.get(field)
        if value is not None:
            payload[field] = utc_to_epoch(value)

    for field in ("graph_stats", "quality_report", "debug_trace", "result", "partial_graph"):
        if field in payload:
            payload[field] = normalize_for_bson(payload[field])
    return payload


def _matched_count(result: Any) -> int:
    return int(getattr(result, "matched_count", 0) or 0)


def _status_str(value: Any) -> str:
    if isinstance(value, PipelineStatus):
        return value.value
    return str(value)


def _is_cancel_terminal_write(job: PipelineJob) -> bool:
    """Cancellation to CANCELLED remains allowed while cancel_requested is true."""
    return job.status is PipelineStatus.CANCELLED


def _worker_save_cas_filter(job: PipelineJob) -> dict[str, Any]:
    """CAS filter for worker saves: generation + non-terminal (+ cancel wins)."""
    cas_filter: dict[str, Any] = {
        "job_id": job.job_id,
        "status": {"$in": _NON_TERMINAL_STATUSES},
        "retry_count": job.retry_count,
    }
    # Non-cancel worker saves (including COMPLETED/FAILED) must not match when
    # cancel was already requested. $max alone is insufficient: it would allow
    # COMPLETED+cancel_requested=true.
    if not _is_cancel_terminal_write(job):
        cas_filter["cancel_requested"] = {"$ne": True}
    return cas_filter


def _classify_save_cas_miss(
    *,
    job: PipelineJob,
    existing_status: str | None,
    existing_retry_count: int | None,
    existing_cancel_requested: bool,
) -> SaveJobOutcome:
    """Classify a CAS miss from a minimal projection of the persisted document."""
    if existing_status is None:
        # Document vanished mid-flight; treat as stale stop (do not invent success).
        return SaveJobOutcome.STALE_GENERATION
    # Terminal wins classification first so a cancelled/completed doc is not
    # misreported as an active CANCEL_REQUESTED for a late progress write.
    if existing_status in _TERMINAL_STATUSES:
        return SaveJobOutcome.ALREADY_TERMINAL
    if not _is_cancel_terminal_write(job) and existing_cancel_requested:
        return SaveJobOutcome.CANCEL_REQUESTED
    if existing_retry_count is not None and int(existing_retry_count) != int(job.retry_count):
        return SaveJobOutcome.STALE_GENERATION
    # Same generation still non-terminal but filter missed (race / concurrent write).
    return SaveJobOutcome.STALE_GENERATION


async def _read_save_cas_projection(job_id: str) -> tuple[str | None, int | None, bool]:
    """Minimal projection for CAS-miss classification."""
    existing = await PipelineJobDocument.find_one(PipelineJobDocument.job_id == job_id)
    if existing is None:
        return None, None, False
    return (
        _status_str(existing.status),
        int(existing.retry_count or 0),
        bool(existing.cancel_requested),
    )


async def save_pipeline_job(job: PipelineJob) -> SaveJobOutcome:
    """Persist a pipeline job with generation-scoped CAS and cancel-wins finalization.

    Worker non-terminal/progress and terminal transitions only match documents that
    are still non-terminal with the same ``retry_count``. Non-cancel writes also
    require ``cancel_requested != true`` so COMPLETED cannot race past cancel.

    Outcomes (never ambiguous bool / false success):
    - APPLIED: CAS update matched
    - CANCEL_REQUESTED: persisted cancel flag; caller must cooperative-cancel
    - STALE_GENERATION: retry_count mismatch or concurrent race
    - ALREADY_TERMINAL: document already terminal

    ``cancel_requested`` uses ``$max`` so progress cannot clear cancellation.
    Cancellation transition to CANCELLED is allowed while the flag is true.
    Infrastructure failures propagate (never false success).
    """
    payload = pipeline_job_to_document(job)
    update_ops = _pipeline_job_update_ops(payload)
    cas_filter = _worker_save_cas_filter(job)

    result = await PipelineJobDocument.find_one(cas_filter).update(update_ops)
    if _matched_count(result):
        return SaveJobOutcome.APPLIED

    status, retry_count, cancel_requested = await _read_save_cas_projection(job.job_id)
    return _classify_save_cas_miss(
        job=job,
        existing_status=status,
        existing_retry_count=retry_count,
        existing_cancel_requested=cancel_requested,
    )


async def create_pipeline_job(job: PipelineJob) -> CreateJobOutcome:
    """Insert a new job without ever mutating a duplicate-key winner.

    Job identity, owner, and source are immutable after this boundary. A true
    UUID collision is returned to the application so it can generate a fresh
    identifier within a bounded retry loop.
    """
    try:
        await PipelineJobDocument(**pipeline_job_to_document(job)).insert()
    except DuplicateKeyError:
        return CreateJobOutcome.COLLISION
    return CreateJobOutcome.CREATED


async def compensate_failed_retry_reschedule(
    job_id: str,
    user_id: str,
    *,
    retry_count: int,
    retryable: bool,
) -> RetryCompensationResult:
    """CAS-compensate a specific post-retry QUEUED generation to FAILED retryable.

    Matches only ``job_id + owner + QUEUED + exact retry_count`` so a newer
    generation or terminal state is never overwritten.
    """
    now = utc_now()
    updated = await PipelineJobDocument.find_one(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": PipelineStatus.QUEUED.value,
            "retry_count": retry_count,
            "cancel_requested": {"$ne": True},
        }
    ).update(
        {
            "$set": {
                "status": PipelineStatus.FAILED.value,
                "current_step": "Retry scheduling failed.",
                "error_code": _RETRY_RESCHEDULE_FAILED_CODE,
                "error_message": _RETRY_RESCHEDULE_FAILED_ERROR,
                "user_message": _RETRY_RESCHEDULE_FAILED_USER_MESSAGE,
                "retryable": retryable,
                "updated_at": now,
                "heartbeat_at": now,
                "completed_at": now,
            }
        },
        response_type=UpdateResponse.NEW_DOCUMENT,
    )
    if updated is not None:
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.APPLIED,
            status=PipelineStatus.FAILED.value,
            retry_count=retry_count,
            cancel_requested=False,
        )

    existing = await load_pipeline_job_for_user(job_id, user_id)
    if existing is None:
        return RetryCompensationResult(outcome=RetryCompensationOutcome.NOT_FOUND)

    persisted_generation = int(existing.get("retry_count") or 0)
    persisted_status = str(existing.get("status") or "")
    cancel_requested = bool(existing.get("cancel_requested"))
    if persisted_generation != retry_count:
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.STALE_GENERATION,
            status=persisted_status,
            retry_count=persisted_generation,
            cancel_requested=cancel_requested,
        )
    if cancel_requested:
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.CANCEL_REQUESTED,
            status=persisted_status,
            retry_count=persisted_generation,
            cancel_requested=True,
        )
    if persisted_status in _TERMINAL_STATUSES:
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.ALREADY_TERMINAL,
            status=persisted_status,
            retry_count=persisted_generation,
            cancel_requested=False,
        )
    if persisted_status != PipelineStatus.QUEUED.value:
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.WORKER_STARTED,
            status=persisted_status,
            retry_count=persisted_generation,
            cancel_requested=False,
        )
    return RetryCompensationResult(
        outcome=RetryCompensationOutcome.CONFLICT,
        status=persisted_status,
        retry_count=persisted_generation,
        cancel_requested=False,
    )


async def load_pipeline_job(job_id: str) -> dict[str, Any] | None:
    """Load one job by id. ``None`` only for genuine absence; DB errors propagate."""
    doc = await PipelineJobDocument.find_one(PipelineJobDocument.job_id == job_id)
    return None if doc is None else _document_to_runtime_payload(doc)


async def load_pipeline_job_cancel_requested(job_id: str) -> bool:
    """Lightweight projection read of just the cancel flag (hot path).

    Infrastructure failures propagate so cooperative cancel never treats an
    outage as ``cancel_requested=false``.
    """
    row = await PipelineJobDocument.find_one(
        PipelineJobDocument.job_id == job_id,
        projection_model=PipelineJobCancelProjection,
    )
    return bool(row.cancel_requested) if row else False


async def load_pipeline_job_for_user(job_id: str, user_id: str) -> dict[str, Any] | None:
    """Owner-scoped load. ``None`` only for absence/ownership miss; DB errors propagate."""
    doc = await PipelineJobDocument.find_one(
        PipelineJobDocument.job_id == job_id,
        PipelineJobDocument.user_id == user_id,
    )
    return None if doc is None else _document_to_runtime_payload(doc)


async def request_cancel_pipeline_job_for_user(job_id: str, user_id: str) -> CancelJobResult:
    """Owner-scoped atomic cancel flag without load-full-save.

    - Non-terminal match → set ``cancel_requested=true`` (idempotent if already true)
    - Owned terminal document → already_terminal (no mutation)
    - Missing / wrong owner → not_found
    Infrastructure failures propagate.
    """
    now = utc_now()
    updated = await PipelineJobDocument.find_one(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": {"$in": _NON_TERMINAL_STATUSES},
        }
    ).update(
        {"$set": {"cancel_requested": True, "updated_at": now}},
        response_type=UpdateResponse.NEW_DOCUMENT,
    )
    if updated is not None:
        status = (
            updated.status.value
            if isinstance(updated.status, PipelineStatus)
            else str(updated.status)
        )
        return CancelJobResult(
            outcome=CancelJobOutcome.REQUESTED,
            status=status,
            cancel_requested=True,
        )

    # No non-terminal match: ownership miss, or already terminal.
    existing = await PipelineJobDocument.find_one(
        PipelineJobDocument.job_id == job_id,
        PipelineJobDocument.user_id == user_id,
        projection_model=PipelineJobStatusProjection,
    )
    if existing is None:
        return CancelJobResult(outcome=CancelJobOutcome.NOT_FOUND)
    status = (
        existing.status.value
        if isinstance(existing.status, PipelineStatus)
        else str(existing.status)
    )
    if status in _TERMINAL_STATUSES:
        return CancelJobResult(
            outcome=CancelJobOutcome.ALREADY_TERMINAL,
            status=status,
            cancel_requested=False,
        )
    # Non-terminal race (update lost to concurrent cancel/status change): re-read flag.
    flag_row = await PipelineJobDocument.find_one(
        PipelineJobDocument.job_id == job_id,
        PipelineJobDocument.user_id == user_id,
        projection_model=PipelineJobCancelProjection,
    )
    if flag_row is not None and bool(flag_row.cancel_requested):
        return CancelJobResult(
            outcome=CancelJobOutcome.REQUESTED,
            status=status,
            cancel_requested=True,
        )
    return CancelJobResult(
        outcome=CancelJobOutcome.CONFLICT,
        status=status,
        cancel_requested=False,
    )


async def transition_pipeline_job_for_retry(
    job_id: str,
    user_id: str,
    *,
    max_retry_count: int,
) -> RetryJobResult:
    """Atomic owner-scoped retry from allowed terminal+retryable state.

    Only this transition clears ``cancel_requested``. It also ``$inc`` retry_count
    under a CAS predicate so a concurrent stale worker (old generation) cannot
    interleave illegal progress over the new run.
    """
    now = utc_now()
    updated = await PipelineJobDocument.find_one(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": {"$in": _RETRYABLE_STATUSES},
            "retryable": True,
            "retry_count": {"$lt": max_retry_count},
            "source_s3_key": {"$type": "string", "$ne": ""},
        }
    ).update(
        {
            "$set": {
                "status": PipelineStatus.QUEUED.value,
                "current_step": "Queued for retry",
                "progress": 0.0,
                "error_message": None,
                "error_code": None,
                "user_message": None,
                "retryable": False,
                "quality_report": None,
                "debug_trace": [],
                "cancel_requested": False,
                "completed_at": None,
                "updated_at": now,
                "heartbeat_at": now,
            },
            "$inc": {"retry_count": 1},
        },
        response_type=UpdateResponse.NEW_DOCUMENT,
    )
    if updated is not None:
        return RetryJobResult(
            outcome=RetryJobOutcome.RETRIED,
            job=_document_to_runtime_payload(updated),
        )

    existing = await load_pipeline_job_for_user(job_id, user_id)
    if existing is None:
        return RetryJobResult(outcome=RetryJobOutcome.NOT_FOUND)

    status = existing.get("status")
    if status not in _RETRYABLE_STATUSES:
        return RetryJobResult(outcome=RetryJobOutcome.INVALID_STATE, job=existing)
    if not existing.get("retryable"):
        return RetryJobResult(outcome=RetryJobOutcome.NOT_RETRYABLE, job=existing)
    if int(existing.get("retry_count") or 0) >= max_retry_count:
        return RetryJobResult(outcome=RetryJobOutcome.MAX_RETRIES, job=existing)
    if not existing.get("source_s3_key"):
        return RetryJobResult(outcome=RetryJobOutcome.NO_SOURCE, job=existing)
    # Eligible but CAS lost (concurrent retry) — surface as invalid/conflict.
    return RetryJobResult(outcome=RetryJobOutcome.INVALID_STATE, job=existing)


async def load_pipeline_job_status_for_user(
    job_id: str,
    user_id: str,
    *,
    include_debug: bool = False,
) -> dict[str, Any] | None:
    if include_debug:
        rows = await PipelineJobDocument.aggregate(
            [
                {"$match": {"job_id": job_id, "user_id": user_id}},
                {
                    "$set": {
                        "result.concept_embedding_count": {
                            "$size": {"$ifNull": ["$result.concept_embeddings", []]}
                        }
                    }
                },
                {"$unset": "result.concept_embeddings"},
                {"$limit": 1},
            ]
        ).to_list()
        return None if not rows else _status_row_to_runtime_payload(rows[0])

    row = await PipelineJobDocument.find_one(
        PipelineJobDocument.job_id == job_id,
        PipelineJobDocument.user_id == user_id,
        projection_model=PipelineJobStatusProjection,
    )
    return None if row is None else _status_row_to_runtime_payload(row)


async def load_many_pipeline_jobs_for_user(
    job_ids: list[str],
    user_id: str,
    *,
    summary_only: bool = False,
) -> dict[str, dict[str, Any]]:
    if not job_ids:
        return {}
    if summary_only:
        rows = await PipelineJobDocument.find(
            {"job_id": {"$in": job_ids}, "user_id": user_id},
            projection_model=PipelineJobLookupProjection,
        ).to_list()
        return {
            row.job_id: {
                "job_id": row.job_id,
                "filename": row.filename,
                "subject_id": row.subject_id,
            }
            for row in rows
        }
    docs = await PipelineJobDocument.find(
        {"job_id": {"$in": job_ids}, "user_id": user_id}
    ).to_list()
    return {doc.job_id: _document_to_runtime_payload(doc) for doc in docs}


async def list_recent_pipeline_jobs(
    *,
    limit: int = 20,
    user_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    filters: list[Any] = []
    if user_id is not None:
        filters.append(PipelineJobDocument.user_id == user_id)
    if status is not None:
        try:
            filters.append(PipelineJobDocument.status == PipelineStatus(status))
        except ValueError:
            filters.append(PipelineJobDocument.status == status)
    rows = await (
        PipelineJobDocument.find(
            *filters,
            projection_model=PipelineJobListProjection,
        )
        .sort(("completed_at", SortDirection.DESCENDING))
        .limit(limit)
        .to_list()
    )

    return [
        {
            "job_id": row.job_id,
            "filename": row.filename,
            "subject_id": row.subject_id,
            "status": row.status.value,
            "page_batch_size": row.page_batch_size,
            "batch_count": row.batch_count,
            "failed_batch_count": row.failed_batch_count,
            "partial_success": row.partial_success,
            "concepts_extracted": row.concepts_extracted,
            "concepts_after_merge": row.concepts_after_merge,
            "relations_verified": row.relations_verified,
            "completed_at": utc_to_epoch(row.completed_at, default=0.0),
        }
        for row in rows
    ]


async def list_active_pipeline_jobs(
    *,
    user_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Full runtime payloads for non-terminal jobs (reaper/recovery scans)."""
    query: dict[str, Any] = {"status": {"$in": _NON_TERMINAL_STATUSES}}
    if user_id is not None:
        query["user_id"] = user_id
    docs = await PipelineJobDocument.find(query).limit(limit).to_list()
    return [_document_to_runtime_payload(doc) for doc in docs]


async def find_recent_active_job_by_source(
    *,
    user_id: str,
    source_s3_key: str,
    window_sec: int,
) -> dict[str, Any] | None:
    """Most recent non-terminal job for this user+source within the window.

    Used to dedup rapid double-submits of the same upload. Returns the
    runtime payload of the newest match, or ``None`` if none exists.
    """
    cutoff = utc_now() - timedelta(seconds=window_sec)
    query: dict[str, Any] = {
        "user_id": user_id,
        "source_s3_key": source_s3_key,
        "status": {"$in": _NON_TERMINAL_STATUSES},
        "created_at": {"$gte": cutoff},
    }
    doc = await (
        PipelineJobDocument.find(query)
        .sort(("created_at", SortDirection.DESCENDING))
        .first_or_none()
    )
    return None if doc is None else _document_to_runtime_payload(doc)


async def list_recent_pipeline_jobs_all_status(
    *,
    user_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Library listing: recent jobs of all statuses for one user."""
    rows = await (
        PipelineJobDocument.find(
            PipelineJobDocument.user_id == user_id,
            projection_model=PipelineJobActiveProjection,
        )
        .sort(("updated_at", SortDirection.DESCENDING))
        .limit(limit)
        .to_list()
    )
    return [
        {
            "job_id": r.job_id,
            "filename": r.filename,
            "subject_id": r.subject_id,
            "status": r.status.value,
            "current_step": r.current_step,
            "progress": r.progress,
            "page_batch_size": r.page_batch_size,
            "batch_count": r.batch_count,
            "failed_batch_count": r.failed_batch_count,
            "partial_success": r.partial_success,
            "concepts_extracted": r.concepts_extracted,
            "concepts_after_merge": r.concepts_after_merge,
            "relations_verified": r.relations_verified,
            "quality_report": normalize_for_bson(r.quality_report),
            "error_code": r.error_code,
            "user_message": r.user_message,
            "retryable": r.retryable,
            "retry_count": r.retry_count,
            "eta_seconds": r.eta_seconds,
            "created_at": utc_to_epoch(r.created_at, default=0.0),
            "updated_at": utc_to_epoch(r.updated_at, default=0.0),
            "heartbeat_at": utc_to_epoch(r.heartbeat_at, default=0.0),
            "completed_at": utc_to_epoch(r.completed_at, default=0.0) if r.completed_at else None,
        }
        for r in rows
    ]


async def delete_pipeline_job_for_user(
    job_id: str,
    user_id: str,
    *,
    delete_sessions: bool = True,
) -> dict[str, Any]:
    """Delete one owner-scoped pipeline job.

    Returns ``deleted_job: 0`` only for genuine absence / ownership miss.
    Database and infrastructure failures propagate; callers must not treat them
    as not-found.
    """
    async with mongo_store.start_session() as session, await session.start_transaction():
        job = await PipelineJobDocument.find_one(
            PipelineJobDocument.job_id == job_id,
            PipelineJobDocument.user_id == user_id,
            session=session,
        )
        if job is None:
            return {"deleted_job": 0, "deleted_sessions": 0}

        deleted_sessions = 0
        if delete_sessions:
            deleted_sessions = await delete_subject_progress_for_user(
                job_id,
                user_id,
                session=session,
            )
        await delete_chunks_for_job(job_id, session=session)
        await job.delete(session=session)
        return {"deleted_job": 1, "deleted_sessions": deleted_sessions}
