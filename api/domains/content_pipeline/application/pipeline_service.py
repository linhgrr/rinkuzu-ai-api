"""Lifecycle orchestration for content pipeline jobs.

This service owns request-independent job scheduling and persistence.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from api.config import get_settings
from api.domains.content_pipeline.application.eta import estimate_eta_seconds
from api.domains.content_pipeline.domain.errors import (
    PipelineJobIdCollisionError,
    PipelineSchedulingUnavailableError,
)
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus
from api.domains.content_pipeline.domain.transitions import CreateJobOutcome

from .ports import CreateJobFn, SaveJobFn, raise_for_save_outcome

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

logger = logging.getLogger(__name__)

_STALLED_USER_MESSAGE = "Processing stalled and was stopped. You can retry."
_MAX_JOB_ID_CREATE_ATTEMPTS = 3


class RunPipelineFn(Protocol):
    def __call__(
        self,
        job: PipelineJob,
        file_path: str,
        prs_threshold: float | None,
        min_confidence: float,
        *,
        apply_reduction: bool,
        page_batch_size: int,
    ) -> Coroutine[Any, Any, None]:
        raise NotImplementedError


class ListActiveJobsFn(Protocol):
    def __call__(self, *, user_id: str | None) -> Coroutine[Any, Any, list[dict[str, Any]]]:
        raise NotImplementedError


class DownloadSourceFn(Protocol):
    def __call__(self, s3_key: str, dest_dir: str) -> Coroutine[Any, Any, str]:
        raise NotImplementedError


class PipelineService:
    """Owns top-level job lifecycle concerns for the content pipeline."""

    def __init__(
        self,
        create_job: CreateJobFn,
        save_job: SaveJobFn,
        run_pipeline: RunPipelineFn,
        *,
        max_concurrent_jobs: int = 2,
        upload_dir: Path | None = None,
    ):
        self._create_job = create_job
        self._save_job = save_job
        self._run_pipeline = run_pipeline
        self._scheduled_tasks: dict[str, asyncio.Task[None]] = {}
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._is_shutting_down = False
        self._upload_dir = upload_dir or (
            Path(__file__).parent.parent.parent.parent.parent / "uploads"
        )

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
        job.eta_seconds = estimate_eta_seconds(
            job, secs_per_page=get_settings().content_pipeline_extraction_secs_per_page
        )
        job.updated_at = now
        job.heartbeat_at = now
        if status.is_terminal:
            job.completed_at = job.completed_at or now
        outcome = await self._save_job(job)
        raise_for_save_outcome(
            job,
            outcome,
            operation=f"persisting status={status.value}",
        )

    async def start_job(
        self,
        *,
        file_path: str,
        subject_id: str | None,
        prs_threshold: float | None,
        min_confidence: float,
        apply_reduction: bool,
        user_id: str | None,
        content_processor_available: bool,
        content_processor_src: str,
        page_batch_size: int,
        source_s3_key: str | None = None,
        dedup_window_sec: int = 0,
        find_recent_duplicate: Callable[[str, str, int], Awaitable[dict[str, Any] | None]]
        | None = None,
    ) -> PipelineJob:
        if self._is_shutting_down:
            raise RuntimeError("Content pipeline is shutting down and cannot accept new jobs.")

        if source_s3_key and find_recent_duplicate is not None and dedup_window_sec > 0:
            existing = await find_recent_duplicate(user_id or "", source_s3_key, dedup_window_sec)
            if isinstance(existing, dict):
                logger.info(
                    "Dedup hit: reusing existing job %s for source %s",
                    existing.get("job_id"),
                    source_s3_key,
                )
                return self.build_job_from_payload(existing)

        job: PipelineJob | None = None
        for _attempt in range(_MAX_JOB_ID_CREATE_ATTEMPTS):
            candidate = self._build_job(
                file_path=file_path,
                subject_id=subject_id,
                user_id=user_id,
                page_batch_size=page_batch_size,
            )
            candidate.source_s3_key = source_s3_key
            candidate.prs_threshold = prs_threshold
            candidate.min_confidence = min_confidence
            candidate.apply_reduction = apply_reduction

            if not content_processor_available:
                candidate.error_code = "pipeline_unavailable"
                candidate.user_message = (
                    "Processing is temporarily unavailable. Please try again later."
                )
                candidate.mark_failed(
                    "Content pipeline modules not available. "
                    f"Expected runtime root: {content_processor_src}"
                )
                return candidate

            outcome = await self._create_job(candidate)
            if outcome is CreateJobOutcome.CREATED:
                job = candidate
                break

        if job is None:
            raise PipelineJobIdCollisionError(
                f"Could not allocate a unique pipeline job id after "
                f"{_MAX_JOB_ID_CREATE_ATTEMPTS} attempts"
            )

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
            job_id=str(uuid.uuid4()),
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
        prs_threshold: float | None,
        min_confidence: float,
        apply_reduction: bool,
        page_batch_size: int,
    ) -> None:
        semaphore = self._concurrency_semaphore

        async def _gated_run() -> None:
            async with semaphore:
                await self._run_pipeline(
                    job,
                    file_path,
                    prs_threshold,
                    min_confidence,
                    apply_reduction=apply_reduction,
                    page_batch_size=page_batch_size,
                )

        task: asyncio.Task[None] = asyncio.create_task(_gated_run())
        self._scheduled_tasks[job.job_id] = task

        def _discard(_t: asyncio.Task[None], jid: str = job.job_id) -> None:
            self._scheduled_tasks.pop(jid, None)

        task.add_done_callback(_discard)

    def build_job_from_payload(self, doc: dict[str, Any]) -> PipelineJob:
        """Rehydrate a :class:`PipelineJob` from a persisted document.

        Symmetric with the repository serialization; tolerant of partial
        payloads via ``.get`` defaults so callers may pass projections.
        """
        job = PipelineJob(
            job_id=doc["job_id"],
            filename=doc["filename"],
            subject_id=doc["subject_id"],
            user_id=doc.get("user_id"),
            status=PipelineStatus(doc["status"]),
            current_step=doc.get("current_step", ""),
            progress=doc.get("progress", 0.0),
            total_chunks=doc.get("total_chunks", 0),
            total_pages=doc.get("total_pages", 0),
            page_batch_size=doc.get("page_batch_size", 10),
            batch_count=doc.get("batch_count", 0),
            failed_batch_count=doc.get("failed_batch_count", 0),
            partial_success=doc.get("partial_success", False),
            concepts_extracted=doc.get("concepts_extracted", 0),
            concepts_after_merge=doc.get("concepts_after_merge", 0),
            relations_verified=doc.get("relations_verified", 0),
            graph_stats=doc.get("graph_stats") or {},
            quality_report=doc.get("quality_report"),
            debug_trace=doc.get("debug_trace") or [],
            result=doc.get("result"),
            partial_graph=doc.get("partial_graph"),
            error_message=doc.get("error_message"),
            error_code=doc.get("error_code"),
            user_message=doc.get("user_message"),
            retryable=doc.get("retryable", False),
            retry_count=doc.get("retry_count", 0),
            cancel_requested=doc.get("cancel_requested", False),
            eta_seconds=doc.get("eta_seconds"),
            source_s3_key=doc.get("source_s3_key"),
            prs_threshold=doc.get("prs_threshold"),
            min_confidence=doc.get("min_confidence", 0.6),
            apply_reduction=doc.get("apply_reduction", True),
        )
        if "created_at" in doc and doc["created_at"] is not None:
            job.created_at = doc["created_at"]
        if "updated_at" in doc and doc["updated_at"] is not None:
            job.updated_at = doc["updated_at"]
        if "heartbeat_at" in doc and doc["heartbeat_at"] is not None:
            job.heartbeat_at = doc["heartbeat_at"]
        if doc.get("completed_at") is not None:
            job.completed_at = doc["completed_at"]
        return job

    async def _fail_as_stalled(self, job: PipelineJob, step: str) -> None:
        job.error_code = "pipeline_stalled"
        job.user_message = _STALLED_USER_MESSAGE
        job.retryable = True
        await self.persist_job_state(job, PipelineStatus.FAILED, step, job.progress)

    def _cancel_local_task(self, job_id: str) -> bool:
        """Best-effort cancel of the in-process background task for ``job_id``.

        Single-instance deployment: the task lives in THIS process, so cancelling
        the local task is the correct lever. In a multi-worker future the task may
        live in another worker and this is a harmless no-op; the Mongo FAILED write
        plus the runner's own cancel/heartbeat checks remain the backstop.
        """
        task = self._scheduled_tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    async def reap_stalled_jobs(
        self,
        *,
        list_active: ListActiveJobsFn,
        stalled_after_sec: float,
    ) -> int:
        """Fail active jobs whose heartbeat has gone stale; return count reaped."""
        now = time.time()
        reaped = 0
        for doc in await list_active(user_id=None):
            heartbeat_at = doc.get("heartbeat_at")
            if heartbeat_at is None or now - heartbeat_at < stalled_after_sec:
                continue
            job = self.build_job_from_payload(doc)
            if self._cancel_local_task(job.job_id):
                logger.info("Cancelled local task for stalled job %s", job.job_id)
            await self._fail_as_stalled(job, "Stalled — reaped")
            reaped += 1
        if reaped:
            logger.warning("Reaped %d stalled pipeline job(s)", reaped)
        return reaped

    async def recover_interrupted_jobs(
        self,
        *,
        list_active: ListActiveJobsFn,
        download_source: DownloadSourceFn,
        recovery_max_age_sec: float,
    ) -> None:
        """Reschedule recent interrupted jobs from S3; fail old/sourceless ones."""
        now = time.time()
        for doc in await list_active(user_id=None):
            job = self.build_job_from_payload(doc)
            too_old = job.created_at and now - job.created_at >= recovery_max_age_sec
            if too_old or not job.source_s3_key:
                await self._fail_as_stalled(job, "Interrupted — not recoverable")
                continue
            await self._reschedule_from_source(job, download_source)

    async def reschedule_retried_job(
        self,
        job: PipelineJob,
        *,
        download_source: DownloadSourceFn,
    ) -> None:
        """Schedule work after an authorized repository retry transition."""
        if not job.source_s3_key:
            raise RuntimeError("Job has no source to retry from")
        await self._reschedule_from_source(job, download_source)

    async def _reschedule_from_source(
        self,
        job: PipelineJob,
        download_source: DownloadSourceFn,
    ) -> None:
        if self._is_shutting_down:
            raise PipelineSchedulingUnavailableError(
                "Content pipeline is shutting down and cannot reschedule jobs."
            )
        if not job.source_s3_key:
            raise ValueError("Job has no source to reschedule from")

        file_path: str | None = None
        ownership_transferred = False
        try:
            file_path = await download_source(job.source_s3_key, str(self._upload_dir))
            await self.persist_job_state(
                job,
                PipelineStatus.QUEUED,
                "Queued for processing",
                PipelineProgress.INIT,
            )
            self._schedule_background_run(
                job,
                file_path=file_path,
                prs_threshold=job.prs_threshold,
                min_confidence=job.min_confidence,
                apply_reduction=job.apply_reduction,
                page_batch_size=job.page_batch_size,
            )
            ownership_transferred = True
            # Yield control so the freshly-scheduled background task can begin.
            await asyncio.sleep(0)
        finally:
            if file_path is not None and not ownership_transferred:
                with contextlib.suppress(OSError):
                    await asyncio.to_thread(Path(file_path).unlink, missing_ok=True)

    async def shutdown(
        self,
        *,
        cancel_running: bool = True,
        timeout_sec: float = 5.0,
    ) -> None:
        """Stop accepting new jobs and drain/cancel scheduled background tasks."""
        self._is_shutting_down = True
        pending_tasks = list(self._scheduled_tasks.values())
        if not pending_tasks:
            return

        if cancel_running:
            for task in pending_tasks:
                task.cancel()

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*pending_tasks, return_exceptions=True),
                timeout=timeout_sec,
            )
        self._scheduled_tasks.clear()
