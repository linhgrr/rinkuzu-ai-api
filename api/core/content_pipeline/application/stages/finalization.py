"""Terminal persistence and writeback helpers for the content pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable

from loguru import logger

from ...domain.jobs import PipelineJob, PipelineStatus


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]
SaveJobFn = Callable[[PipelineJob], Awaitable[bool]]


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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=bucket_name,
                Key=cache_key,
                Body=cache_data,
                ContentType="application/json",
            ),
        )
        logger.info(f"[Pipeline] Uploaded result to S3 cache {cache_key}")
    except Exception as exc:
        logger.warning(f"[Pipeline] Failed to save S3 cache: {exc}")


async def persist_terminal_failure(
    job: PipelineJob,
    *,
    error: Exception,
    save_job: SaveJobFn,
) -> None:
    """Persist a failed terminal job state without raising secondary errors."""
    job.error_message = str(error)
    logger.error(f"[Pipeline] Job {job.job_id} failed: {error}")
    job.status = PipelineStatus.FAILED
    job.current_step = f"Error: {error}"
    try:
        saved = await save_job(job)
        if not saved:
            logger.error(f"[Pipeline] Failed to persist terminal failure state for job {job.job_id}")
    except Exception as persist_exc:
        logger.error(
            f"[Pipeline] Failed to persist terminal failure state for job {job.job_id}: {persist_exc}"
        )
