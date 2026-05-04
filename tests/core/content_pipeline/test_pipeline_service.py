import pytest

from api.core.content_pipeline.application.pipeline_service import PipelineService
from api.core.content_pipeline.domain.jobs import PipelineStatus


@pytest.mark.anyio
async def test_start_job_persists_pending_then_queued_and_schedules_background_task():
    saved_statuses: list[str] = []
    scheduled_calls = []

    async def save_job(job):
        saved_statuses.append(job.status.value)
        return True

    async def run_pipeline(job, file_path, prs_threshold, min_confidence, apply_reduction):
        raise AssertionError("background task should not run during this test")

    service = PipelineService(save_job=save_job, run_pipeline=run_pipeline)
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
    )

    assert job.filename == "algebra.pdf"
    assert job.subject_id == "algebra"
    assert job.status == PipelineStatus.QUEUED
    assert saved_statuses == ["pending", "queued"]
    assert len(scheduled_calls) == 1


@pytest.mark.anyio
async def test_start_job_returns_failed_job_when_content_processor_is_unavailable():
    async def save_job(job):
        raise AssertionError("save_job should not be called when dependencies are unavailable")

    async def run_pipeline(job, file_path, prs_threshold, min_confidence, apply_reduction):
        raise AssertionError("run_pipeline should not be called in this test")

    service = PipelineService(save_job=save_job, run_pipeline=run_pipeline)

    job = await service.start_job(
        file_path="fixtures/algebra.pdf",
        subject_id="math",
        prs_threshold=0.75,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="user-1",
        content_processor_available=False,
        content_processor_src="fixtures/content-pipeline-runtime",
    )

    assert job.status == PipelineStatus.FAILED
    assert job.error_code == "pipeline_unavailable"
    assert job.retryable is False
    assert "Content pipeline modules not available" in (job.error_message or "")
