"""Terminal persistence and writeback helpers for the content pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import time

from loguru import logger

from api.core.content_pipeline.domain.errors import PipelineStageTimeoutError
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from .execution import run_blocking_stage

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]
SaveJobFn = Callable[[PipelineJob], Awaitable[bool]]


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
    job.completed_at = time.time()
    await persist_job_state(job, PipelineStatus.COMPLETED, "Processing complete!", 1.0)
    logger.info(f"[Pipeline] Job {job.job_id} completed: {job.concepts_after_merge} concepts")


async def upload_result_cache(
    *,
    result: dict,
    s3_client,
    bucket_name: str | None,
    cache_key: str | None,
) -> None:
    """Best-effort S3 cache upload for completed pipeline results."""
    if not s3_client or not bucket_name or not cache_key:
        return

    try:
        cache_data = json.dumps(result, ensure_ascii=False)
        await run_blocking_stage(
            s3_client.put_object,
            Bucket=bucket_name,
            Key=cache_key,
            Body=cache_data,
            ContentType="application/json",
            stage_name="s3_cache_upload",
        )
        logger.info(f"[Pipeline] Uploaded result to S3 cache {cache_key}")
    except Exception as exc:
        logger.warning(f"[Pipeline] Failed to save S3 cache: {exc}")


async def persist_terminal_failure(
    job: PipelineJob,
    *,
    error: BaseException,
    save_job: SaveJobFn,
) -> None:
    """Persist a terminal job state without raising secondary errors."""
    details = classify_terminal_failure(job, error)
    job.completed_at = time.time()
    job.status = details.status
    job.error_code = details.error_code
    job.error_message = details.error_message
    job.user_message = details.user_message
    job.retryable = details.retryable
    job.current_step = details.current_step
    logger.error(
        f"[Pipeline] Job {job.job_id} ended with {details.status.value}: {details.error_message}"
    )
    try:
        saved = await save_job(job)
        if not saved:
            logger.error(
                f"[Pipeline] Failed to persist terminal failure state for job {job.job_id}"
            )
    except Exception as persist_exc:
        logger.error(
            f"[Pipeline] Failed to persist terminal failure state for job {job.job_id}: {persist_exc}"
        )


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

    return TerminalFailureDetails(
        status=PipelineStatus.FAILED,
        error_code="pipeline_failed",
        error_message=str(error),
        user_message="We could not finish processing this document. Please try again.",
        retryable=False,
        current_step=f"Error: {error}",
    )
