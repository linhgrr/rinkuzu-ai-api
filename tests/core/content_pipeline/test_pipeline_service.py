import asyncio

import pytest

from api.domains.content_pipeline.application.pipeline_service import PipelineService
from api.domains.content_pipeline.domain.jobs import PipelineStatus
from api.domains.content_pipeline.domain.transitions import CreateJobOutcome, SaveJobOutcome


@pytest.mark.anyio
async def test_start_job_persists_pending_then_queued_and_schedules_background_task():
    saved_statuses: list[str] = []
    scheduled_calls = []

    async def create_job(job):
        saved_statuses.append(job.status.value)
        return CreateJobOutcome.CREATED

    async def save_job(job):
        saved_statuses.append(job.status.value)
        return SaveJobOutcome.APPLIED

    async def run_pipeline(
        job,
        file_path,
        prs_threshold,
        min_confidence,
        *,
        apply_reduction,
        page_batch_size,
    ):
        raise AssertionError("background task should not run during this test")

    service = PipelineService(create_job=create_job, save_job=save_job, run_pipeline=run_pipeline)
    service._schedule_background_run = lambda job, **kwargs: scheduled_calls.append((job, kwargs))

    job = await service.start_job(
        file_path="fixtures/algebra.pdf",
        subject_id=None,
        prs_threshold=0.75,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="user-1",
        content_processor_available=True,
        content_processor_src="fixtures/content-pipeline-runtime",
        page_batch_size=10,
    )

    assert job.filename == "algebra.pdf"
    assert job.subject_id == "algebra"
    assert job.status == PipelineStatus.QUEUED
    assert saved_statuses == ["pending", "queued"]
    assert len(scheduled_calls) == 1


@pytest.mark.anyio
async def test_start_job_returns_failed_job_when_content_processor_is_unavailable():
    async def create_job(job):
        raise AssertionError("create_job should not be called when dependencies are unavailable")

    async def save_job(job):
        raise AssertionError("save_job should not be called when dependencies are unavailable")

    async def run_pipeline(
        job,
        file_path,
        prs_threshold,
        min_confidence,
        *,
        apply_reduction,
        page_batch_size,
    ):
        raise AssertionError("run_pipeline should not be called in this test")

    service = PipelineService(create_job=create_job, save_job=save_job, run_pipeline=run_pipeline)

    job = await service.start_job(
        file_path="fixtures/algebra.pdf",
        subject_id="math",
        prs_threshold=0.75,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="user-1",
        content_processor_available=False,
        content_processor_src="fixtures/content-pipeline-runtime",
        page_batch_size=10,
    )

    assert job.status == PipelineStatus.FAILED
    assert job.error_code == "pipeline_unavailable"
    assert job.retryable is False
    assert "Content pipeline modules not available" in (job.error_message or "")


@pytest.mark.anyio
async def test_shutdown_cancels_inflight_background_tasks():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def create_job(job):
        return CreateJobOutcome.CREATED

    async def save_job(job):
        return SaveJobOutcome.APPLIED

    async def run_pipeline(
        job,
        file_path,
        prs_threshold,
        min_confidence,
        *,
        apply_reduction,
        page_batch_size,
    ):
        del job, file_path, prs_threshold, min_confidence, apply_reduction, page_batch_size
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    service = PipelineService(create_job=create_job, save_job=save_job, run_pipeline=run_pipeline)
    await service.start_job(
        file_path="fixtures/algebra.pdf",
        subject_id=None,
        prs_threshold=0.75,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="user-1",
        content_processor_available=True,
        content_processor_src="fixtures/content-pipeline-runtime",
        page_batch_size=10,
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)
    await service.shutdown(timeout_sec=0.5)

    assert cancelled.is_set()


@pytest.mark.anyio
async def test_start_job_retries_collision_without_mutating_winner():
    created_ids: list[str] = []
    saved_ids: list[str] = []

    async def create_job(job):
        created_ids.append(job.job_id)
        return CreateJobOutcome.COLLISION if len(created_ids) == 1 else CreateJobOutcome.CREATED

    async def save_job(job):
        saved_ids.append(job.job_id)
        return SaveJobOutcome.APPLIED

    async def run_pipeline(*_args, **_kwargs):
        return None

    service = PipelineService(create_job=create_job, save_job=save_job, run_pipeline=run_pipeline)
    service._schedule_background_run = lambda *_args, **_kwargs: None
    job = await service.start_job(
        file_path="fixtures/algebra.pdf",
        subject_id="math",
        prs_threshold=0.75,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="user-1",
        content_processor_available=True,
        content_processor_src="fixtures/content-pipeline-runtime",
        page_batch_size=10,
        source_s3_key="uploads/algebra.pdf",
    )

    assert len(created_ids) == 2
    assert created_ids[0] != created_ids[1]
    assert all(len(value) == 36 for value in created_ids)
    assert saved_ids == [job.job_id]
