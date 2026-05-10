"""Serialization helpers for content pipeline persistence."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from api.core.content_pipeline.domain.jobs import PipelineJob


def _to_bson_safe(value: Any) -> Any:
    """Recursively normalize arbitrary Python objects into BSON-safe values."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _to_bson_safe(value.model_dump())
    if is_dataclass(value):
        return _to_bson_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_bson_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_bson_safe(item) for item in value]
    return value


def pipeline_job_to_document(job: PipelineJob) -> dict[str, Any]:
    """Convert a pipeline job into the Mongo persistence shape.

    The serializer mirrors the current repository contract so this refactor
    stays behavior-preserving while giving future phases a single place to
    evolve the persistence schema.
    """

    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "subject_id": job.subject_id,
        "user_id": job.user_id,
        "status": job.status.value,
        "total_chunks": job.total_chunks,
        "total_pages": job.total_pages,
        "page_batch_size": job.page_batch_size,
        "batch_count": job.batch_count,
        "failed_batch_count": job.failed_batch_count,
        "partial_success": job.partial_success,
        "concepts_extracted": job.concepts_extracted,
        "concepts_after_merge": job.concepts_after_merge,
        "relations_verified": job.relations_verified,
        "graph_stats": _to_bson_safe(job.graph_stats) if isinstance(job.graph_stats, dict) else {},
        "result": _to_bson_safe(job.result),
        "current_step": job.current_step,
        "progress": job.progress,
        "error_message": job.error_message,
        "error_code": job.error_code,
        "user_message": job.user_message,
        "retryable": job.retryable,
        "partial_graph": _to_bson_safe(job.partial_graph),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "heartbeat_at": job.heartbeat_at,
        "completed_at": job.completed_at if job.status.is_terminal else None,
    }
