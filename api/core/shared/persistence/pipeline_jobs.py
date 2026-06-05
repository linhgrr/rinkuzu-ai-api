from __future__ import annotations

from typing import Any

from beanie.odm.enums import SortDirection
from loguru import logger

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.core.shared import mongo_store

from .common import epoch_to_utc, normalize_for_bson, optional_epoch_to_utc, utc_to_epoch
from .document_chunks import delete_chunks_for_job
from .documents import (
    PipelineJobActiveProjection,
    PipelineJobDocument,
    PipelineJobListProjection,
    PipelineJobLookupProjection,
)
from .subject_progress import delete_subject_progress_for_user

_NON_TERMINAL_STATUSES = [s.value for s in PipelineStatus if not s.is_terminal]


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


def _document_to_runtime_payload(doc: PipelineJobDocument) -> dict[str, Any]:
    return {
        "job_id": doc.job_id,
        "filename": doc.filename,
        "subject_id": doc.subject_id,
        "user_id": doc.user_id,
        "status": doc.status.value,
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


async def save_pipeline_job(job: PipelineJob) -> bool:
    try:
        payload = pipeline_job_to_document(job)
        existing = await PipelineJobDocument.find_one(PipelineJobDocument.job_id == job.job_id)
        if existing is None:
            await PipelineJobDocument(**payload).insert()
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
            await existing.replace()
    except Exception:
        logger.exception("[PipelineStore] save failed job_id={}", job.job_id)
        return False
    return True


async def load_pipeline_job(job_id: str) -> dict[str, Any] | None:
    try:
        doc = await PipelineJobDocument.find_one(PipelineJobDocument.job_id == job_id)
    except Exception:
        logger.exception("[PipelineStore] load failed job_id={}", job_id)
        return None
    return None if doc is None else _document_to_runtime_payload(doc)


async def load_pipeline_job_for_user(job_id: str, user_id: str) -> dict[str, Any] | None:
    try:
        doc = await PipelineJobDocument.find_one(
            PipelineJobDocument.job_id == job_id,
            PipelineJobDocument.user_id == user_id,
        )
    except Exception:
        logger.exception(
            "[PipelineStore] load_for_user failed job_id={} user_id={}", job_id, user_id
        )
        return None
    return None if doc is None else _document_to_runtime_payload(doc)


async def load_many_pipeline_jobs_for_user(
    job_ids: list[str],
    user_id: str,
    *,
    summary_only: bool = False,
) -> dict[str, dict[str, Any]]:
    if not job_ids:
        return {}
    try:
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
    except Exception:
        logger.exception("[PipelineStore] load_many_for_user failed user_id={}", user_id)
        return {}
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
    try:
        rows = await (
            PipelineJobDocument.find(
                *filters,
                projection_model=PipelineJobListProjection,
            )
            .sort(("completed_at", SortDirection.DESCENDING))
            .limit(limit)
            .to_list()
        )
    except Exception:
        logger.exception("[PipelineStore] list_recent failed")
        return []

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
    try:
        docs = await PipelineJobDocument.find(query).limit(limit).to_list()
    except Exception:
        logger.exception("[PipelineStore] list_active failed user_id={}", user_id)
        return []
    return [_document_to_runtime_payload(doc) for doc in docs]


async def list_recent_pipeline_jobs_all_status(
    *,
    user_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Library listing: recent jobs of all statuses for one user."""
    try:
        rows = await (
            PipelineJobDocument.find(
                PipelineJobDocument.user_id == user_id,
                projection_model=PipelineJobActiveProjection,
            )
            .sort(("updated_at", SortDirection.DESCENDING))
            .limit(limit)
            .to_list()
        )
    except Exception:
        logger.exception("[PipelineStore] list_all_status failed user_id={}", user_id)
        return []
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
    try:
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
    except Exception:
        logger.exception(
            "[PipelineStore] delete_for_user failed job_id={} user_id={}", job_id, user_id
        )
        return {"deleted_job": 0, "deleted_sessions": 0}
