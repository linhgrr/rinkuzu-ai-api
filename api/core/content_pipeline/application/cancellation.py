"""Cooperative cancellation for content-pipeline jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.core.content_pipeline.domain.jobs import PipelineJob


class JobCancelledError(Exception):
    """Raised at a stage boundary when a job's cancel flag is set."""


def raise_if_cancelled(job: PipelineJob) -> None:
    """Raise JobCancelledError if the job has been flagged for cancellation."""
    if job.cancel_requested:
        raise JobCancelledError(f"Job {job.job_id} cancelled by user")
