"""Terminal persistence and writeback helpers for the content pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any

from loguru import logger

from api.domains.content_pipeline.domain.errors import (
    PipelineQualityGateError,
    PipelineStageTimeoutError,
)
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus
from api.shared.persistence.common import normalize_for_bson

from ..ports import PersistJobStateFn, SaveJobFn  # noqa: TC001
from .execution import run_blocking_stage, safe_run


@dataclass(frozen=True)
class TerminalFailureDetails:
    """Normalized terminal failure payload persisted for job polling."""

    status: PipelineStatus
    error_code: str
    error_message: str
    user_message: str
    retryable: bool
    current_step: str


async def complete_pipeline_job(
    job: PipelineJob,
    *,
    persist_job_state: PersistJobStateFn,
) -> None:
    """Mark a pipeline job completed and persist the terminal state."""
    now = time.time()
    job.completed_at = now
    job.updated_at = now
    job.heartbeat_at = now
    await persist_job_state(
        job, PipelineStatus.COMPLETED, "Processing complete!", PipelineProgress.COMPLETE
    )
    logger.info("[Pipeline] Job {} completed: {} concepts", job.job_id, job.concepts_after_merge)


async def upload_result_cache(
    *,
    result: dict,
    s3_client: Any,
    bucket_name: str | None,
    cache_key: str | None,
) -> None:
    """Best-effort S3 cache upload for completed pipeline results."""
    if not s3_client or not bucket_name or not cache_key:
        return

    async def _upload() -> Any:
        cache_data = json.dumps(normalize_for_bson(result), ensure_ascii=False)
        await run_blocking_stage(
            s3_client.put_object,
            Bucket=bucket_name,
            Key=cache_key,
            Body=cache_data,
            ContentType="application/json",
            stage_name="s3_cache_upload",
        )
        logger.info("[Pipeline] Uploaded result to S3 cache {}", cache_key)

    await safe_run(_upload, fail_message="Failed to save S3 cache")


async def persist_terminal_failure(
    job: PipelineJob,
    *,
    error: BaseException,
    save_job: SaveJobFn,
) -> None:
    """Persist a terminal job state without raising secondary errors."""
    details = classify_terminal_failure(job, error)
    now = time.time()
    job.completed_at = now
    job.updated_at = now
    job.heartbeat_at = now
    job.status = details.status
    job.error_code = details.error_code
    job.error_message = details.error_message
    job.user_message = details.user_message
    job.retryable = details.retryable
    job.current_step = details.current_step
    logger.error(
        "[Pipeline] Job {} ended with {}: {}",
        job.job_id,
        details.status.value,
        details.error_message,
    )
    if not await safe_run(
        lambda: save_job(job),
        fail_message=f"Failed to persist terminal failure state for job {job.job_id}",
        fallback=False,
    ):
        logger.error("[Pipeline] Failed to persist terminal failure state for job {}", job.job_id)


def classify_terminal_failure(job: PipelineJob, error: BaseException) -> TerminalFailureDetails:
    """Map runtime failures into stable terminal job payloads."""
    stage_hint = job.current_step or "the current stage"

    if isinstance(error, asyncio.CancelledError):
        return TerminalFailureDetails(
            status=PipelineStatus.CANCELLED,
            error_code="pipeline_cancelled",
            error_message="Pipeline execution was cancelled.",
            user_message="Processing was interrupted before completion. Please retry.",
            retryable=True,
            current_step="Processing cancelled.",
        )

    if isinstance(error, (PipelineStageTimeoutError, TimeoutError, asyncio.TimeoutError)):
        if isinstance(error, PipelineStageTimeoutError):
            detail = str(error)
            stage_hint = error.stage_name.replace("_", " ")
        else:
            detail = str(error) or f"Pipeline timed out while {stage_hint}"
        return TerminalFailureDetails(
            status=PipelineStatus.FAILED,
            error_code="pipeline_timeout",
            error_message=detail,
            user_message="Processing is taking longer than expected. Please try again.",
            retryable=True,
            current_step=f"Timed out while {stage_hint}.",
        )

    if isinstance(error, PipelineQualityGateError):
        job.quality_report = error.report
        return TerminalFailureDetails(
            status=PipelineStatus.FAILED,
            error_code="pipeline_quality_gate_failed",
            error_message=str(error),
            user_message=(
                "The document was processed, but the generated knowledge graph did not pass "
                "quality checks. Please retry or upload a clearer PDF."
            ),
            retryable=True,
            current_step="Quality checks failed.",
        )

    return TerminalFailureDetails(
        status=PipelineStatus.FAILED,
        error_code="pipeline_failed",
        error_message=str(error),
        user_message="We could not finish processing this document. Please try again.",
        retryable=False,
        current_step=f"Error: {error}",
    )
