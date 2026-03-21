"""Serialization helpers for content pipeline persistence."""

from __future__ import annotations

import time
from typing import Any

from ..domain.jobs import PipelineJob


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
        "concepts_extracted": job.concepts_extracted,
        "concepts_after_merge": job.concepts_after_merge,
        "relations_verified": job.relations_verified,
        "graph_stats": job.graph_stats if isinstance(job.graph_stats, dict) else {},
        "result": job.result,
        "current_step": job.current_step,
        "progress": job.progress,
        "error_message": job.error_message,
        "partial_graph": job.partial_graph,
        "created_at": job.created_at,
        "completed_at": job.completed_at or time.time(),
    }
