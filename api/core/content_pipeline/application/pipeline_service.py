"""Lifecycle orchestration for content pipeline jobs.

This service owns request-independent job scheduling and persistence.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine  # noqa: TC003
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from .ports import SaveJobFn  # noqa: TC001


class RunPipelineFn(Protocol):
    def __call__(
        self,
        job: PipelineJob,
        file_path: str,
        prs_threshold: float,
        min_confidence: float,
        *,
        apply_reduction: bool,
        page_batch_size: int,
    ) -> Coroutine[Any, Any, None]: ...


class PipelineService:
    """Owns top-level job lifecycle concerns for the content pipeline."""

    def __init__(self, save_job: SaveJobFn, run_pipeline: RunPipelineFn):
        self._save_job = save_job
        self._run_pipeline = run_pipeline
        self._scheduled_tasks: set[asyncio.Task[None]] = set()

    async def persist_job_state(
        self,
        job: PipelineJob,
        status: PipelineStatus,
        step: str,
        progress: float,
    ) -> None:
        now = time.time()
        job.status = status
        job.current_step = step
        job.progress = progress
        job.updated_at = now
        job.heartbeat_at = now
        if status.is_terminal:
            job.completed_at = job.completed_at or now
        saved = await self._save_job(job)
        if not saved:
            raise RuntimeError(
                f"Failed to persist pipeline job {job.job_id} at status={status.value}"
            )

    async def start_job(
        self,
        *,
        file_path: str,
        subject_id: str | None,
        prs_threshold: float,
        min_confidence: float,
        apply_reduction: bool,
        user_id: str | None,
        content_processor_available: bool,
        content_processor_src: str,
        page_batch_size: int,
    ) -> PipelineJob:
        job = self._build_job(
            file_path=file_path,
            subject_id=subject_id,
            user_id=user_id,
            page_batch_size=page_batch_size,
        )

        if not content_processor_available:
            job.error_code = "pipeline_unavailable"
            job.user_message = "Processing is temporarily unavailable. Please try again later."
            job.mark_failed(
                "Content pipeline modules not available. "
                f"Expected runtime root: {content_processor_src}"
            )
            return job

        saved = await self._save_job(job)
        if not saved:
            raise RuntimeError(f"Failed to persist pipeline job {job.job_id}")

        await self.persist_job_state(
            job,
            PipelineStatus.QUEUED,
            "Queued for processing",
            PipelineProgress.INIT,
        )

        self._schedule_background_run(
            job,
            file_path=file_path,
            prs_threshold=prs_threshold,
            min_confidence=min_confidence,
            apply_reduction=apply_reduction,
            page_batch_size=page_batch_size,
        )

        return job

    @staticmethod
    def _build_job(
        *,
        file_path: str,
        subject_id: str | None,
        user_id: str | None,
        page_batch_size: int,
    ) -> PipelineJob:
        file_name = Path(file_path).name
        normalized_subject_id = subject_id or Path(file_path).stem
        return PipelineJob(
            job_id=str(uuid.uuid4())[:8],
            filename=file_name,
            subject_id=normalized_subject_id,
            user_id=user_id,
            page_batch_size=page_batch_size,
        )

    def _schedule_background_run(
        self,
        job: PipelineJob,
        *,
        file_path: str,
        prs_threshold: float,
        min_confidence: float,
        apply_reduction: bool,
        page_batch_size: int,
    ) -> None:
        task: asyncio.Task[None] = asyncio.create_task(
            self._run_pipeline(
                job,
                file_path,
                prs_threshold,
                min_confidence,
                apply_reduction=apply_reduction,
                page_batch_size=page_batch_size,
            )
        )
        self._scheduled_tasks.add(task)
        task.add_done_callback(self._scheduled_tasks.discard)
