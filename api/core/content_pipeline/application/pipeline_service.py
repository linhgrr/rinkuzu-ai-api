"""Lifecycle orchestration for content pipeline jobs.

This service intentionally keeps the current external behavior while moving
request-independent job scheduling and persistence out of the legacy
`orchestrator.py` module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
import uuid

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

SaveJobFn = Callable[[PipelineJob], Awaitable[bool]]
RunPipelineFn = Callable[[PipelineJob, str, float, float, bool], Awaitable[None]]


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
        job.status = status
        job.current_step = step
        job.progress = progress
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
    ) -> PipelineJob:
        job = self._build_job(file_path=file_path, subject_id=subject_id, user_id=user_id)

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
            0.01,
        )

        self._schedule_background_run(
            job,
            file_path=file_path,
            prs_threshold=prs_threshold,
            min_confidence=min_confidence,
            apply_reduction=apply_reduction,
        )

        return job

    async def run_job_and_cleanup(
        self,
        job: PipelineJob,
        file_path: str,
        prs_threshold: float,
        min_confidence: float,
        apply_reduction: bool,
    ) -> None:
        try:
            await self._run_pipeline(
                job,
                file_path,
                prs_threshold,
                min_confidence,
                apply_reduction,
            )
        finally:
            self._cleanup_file(file_path)

    @staticmethod
    def _build_job(
        *,
        file_path: str,
        subject_id: str | None,
        user_id: str | None,
    ) -> PipelineJob:
        file_name = Path(file_path).name
        normalized_subject_id = subject_id or Path(file_path).stem
        return PipelineJob(
            job_id=str(uuid.uuid4())[:8],
            filename=file_name,
            subject_id=normalized_subject_id,
            user_id=user_id,
        )

    @staticmethod
    def _cleanup_file(file_path: str) -> None:
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def _schedule_background_run(
        self,
        job: PipelineJob,
        *,
        file_path: str,
        prs_threshold: float,
        min_confidence: float,
        apply_reduction: bool,
    ) -> None:
        task = asyncio.create_task(
            self.run_job_and_cleanup(
                job,
                file_path,
                prs_threshold,
                min_confidence,
                apply_reduction,
            )
        )
        self._scheduled_tasks.add(task)
        task.add_done_callback(self._scheduled_tasks.discard)
