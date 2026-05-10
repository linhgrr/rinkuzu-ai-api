from __future__ import annotations

from typing import Any

from beanie.odm.enums import SortDirection
from loguru import logger

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.core.shared import mongo_store

from .common import epoch_to_utc, normalize_for_bson, optional_epoch_to_utc, utc_to_epoch
from .document_chunks import delete_chunks_for_job
from .documents import (
    PipelineJobDocument,
    PipelineJobListProjection,
    PipelineJobLookupProjection,
)
from .subject_progress import delete_subject_progress_for_user


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
