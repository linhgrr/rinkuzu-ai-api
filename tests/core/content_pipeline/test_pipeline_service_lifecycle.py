import time

import pytest

from api.core.content_pipeline.application.pipeline_service import PipelineService
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def _make_service(saved, run_calls):
    async def save_job(job):
        saved[job.job_id] = job
        return True

    async def run_pipeline(
        job, file_path, prs_threshold, min_confidence, *, apply_reduction, page_batch_size
    ):
        run_calls.append(job.job_id)

    return PipelineService(save_job=save_job, run_pipeline=run_pipeline, max_concurrent_jobs=2)


@pytest.mark.asyncio
async def test_request_cancel_persists_flag():
    saved = {}
    svc = _make_service(saved, [])
    job = PipelineJob(
        job_id="j1", filename="a.pdf", subject_id="a", status=PipelineStatus.EXTRACTING
    )
    await svc.request_cancel(job)
    assert job.cancel_requested is True
    assert saved["j1"].cancel_requested is True


@pytest.mark.asyncio
async def test_reaper_marks_stalled_failed_retryable():
    saved = {}
    svc = _make_service(saved, [])

    async def list_active(**kwargs):
        return [
            {
                "job_id": "j2",
                "filename": "a.pdf",
                "subject_id": "a",
                "user_id": "u",
                "status": "extracting",
                "heartbeat_at": time.time() - 10_000,
                "progress": 0.3,
            }
        ]

    reaped = await svc.reap_stalled_jobs(list_active=list_active, stalled_after_sec=900)
    assert reaped == 1
    assert saved["j2"].status is PipelineStatus.FAILED
    assert saved["j2"].error_code == "pipeline_stalled"
    assert saved["j2"].retryable is True


@pytest.mark.asyncio
async def test_reaper_skips_fresh_jobs():
    saved = {}
    svc = _make_service(saved, [])

    async def list_active(**kwargs):
        return [
            {
                "job_id": "fresh",
                "filename": "a.pdf",
                "subject_id": "a",
                "user_id": "u",
                "status": "extracting",
                "heartbeat_at": time.time(),
                "progress": 0.3,
            }
        ]

    reaped = await svc.reap_stalled_jobs(list_active=list_active, stalled_after_sec=900)
    assert reaped == 0
    assert "fresh" not in saved


@pytest.mark.asyncio
async def test_recovery_reschedules_recent_and_fails_old():
    saved = {}
    run_calls = []
    svc = _make_service(saved, run_calls)
    now = time.time()

    async def list_active(**kwargs):
        return [
            {
                "job_id": "recent",
                "filename": "a.pdf",
                "subject_id": "a",
                "user_id": "u",
                "status": "extracting",
                "created_at": now - 100,
                "heartbeat_at": now - 100,
                "progress": 0.3,
                "source_s3_key": "k1",
                "page_batch_size": 10,
            },
            {
                "job_id": "old",
                "filename": "b.pdf",
                "subject_id": "b",
                "user_id": "u",
                "status": "extracting",
                "created_at": now - 99_999,
                "heartbeat_at": now - 99_999,
                "progress": 0.3,
                "source_s3_key": "k2",
                "page_batch_size": 10,
            },
        ]

    async def download_source(s3_key, dest_dir):
        return f"/tmp/{s3_key}.pdf"

    await svc.recover_interrupted_jobs(
        list_active=list_active, download_source=download_source, recovery_max_age_sec=3600
    )
    assert "recent" in run_calls
    assert saved["old"].status is PipelineStatus.FAILED
    assert saved["old"].error_code == "pipeline_stalled"


@pytest.mark.asyncio
async def test_recovery_fails_job_without_source():
    saved = {}
    run_calls = []
    svc = _make_service(saved, run_calls)
    now = time.time()

    async def list_active(**kwargs):
        return [
            {
                "job_id": "nosrc",
                "filename": "a.pdf",
                "subject_id": "a",
                "user_id": "u",
                "status": "extracting",
                "created_at": now - 10,
                "heartbeat_at": now - 10,
                "progress": 0.1,
                "source_s3_key": None,
                "page_batch_size": 10,
            }
        ]

    async def download_source(s3_key, dest_dir):
        raise AssertionError("should not download when no source")

    await svc.recover_interrupted_jobs(
        list_active=list_active, download_source=download_source, recovery_max_age_sec=3600
    )
    assert saved["nosrc"].status is PipelineStatus.FAILED
    assert "nosrc" not in run_calls


@pytest.mark.asyncio
async def test_retry_job_reschedules_from_source():
    saved = {}
    run_calls = []
    svc = _make_service(saved, run_calls)
    job = PipelineJob(job_id="r1", filename="a.pdf", subject_id="a", status=PipelineStatus.FAILED)
    job.retryable = True
    job.source_s3_key = "k1"

    async def download_source(s3_key, dest_dir):
        return f"/tmp/{s3_key}.pdf"

    await svc.retry_job(job, download_source=download_source, max_retry_count=3)
    assert job.retry_count == 1
    assert job.status is PipelineStatus.QUEUED
    assert "r1" in run_calls


@pytest.mark.asyncio
async def test_retry_job_rejects_non_retryable():
    saved = {}
    svc = _make_service(saved, [])
    job = PipelineJob(job_id="r2", filename="a.pdf", subject_id="a", status=PipelineStatus.FAILED)
    job.retryable = False
    job.source_s3_key = "k"

    async def download_source(s3_key, dest_dir):
        return "/tmp/x.pdf"

    with pytest.raises(RuntimeError):
        await svc.retry_job(job, download_source=download_source, max_retry_count=3)


@pytest.mark.asyncio
async def test_retry_job_rejects_over_limit():
    saved = {}
    svc = _make_service(saved, [])
    job = PipelineJob(job_id="r3", filename="a.pdf", subject_id="a", status=PipelineStatus.FAILED)
    job.retryable = True
    job.source_s3_key = "k"
    job.retry_count = 3

    async def download_source(s3_key, dest_dir):
        return "/tmp/x.pdf"

    with pytest.raises(RuntimeError):
        await svc.retry_job(job, download_source=download_source, max_retry_count=3)


def test_build_job_from_payload_maps_fields():
    saved = {}
    svc = _make_service(saved, [])
    job = svc.build_job_from_payload(
        {
            "job_id": "p1",
            "filename": "a.pdf",
            "subject_id": "a",
            "user_id": "u",
            "status": "extracting",
            "progress": 0.4,
            "total_pages": 12,
            "source_s3_key": "k1",
            "prs_threshold": 0.55,
            "min_confidence": 0.7,
            "apply_reduction": False,
            "retry_count": 2,
            "page_batch_size": 8,
        }
    )
    assert job.job_id == "p1"
    assert job.status is PipelineStatus.EXTRACTING
    assert job.source_s3_key == "k1"
    assert job.prs_threshold == 0.55
    assert job.min_confidence == 0.7
    assert job.apply_reduction is False
    assert job.retry_count == 2
    assert job.page_batch_size == 8
    assert job.total_pages == 12
