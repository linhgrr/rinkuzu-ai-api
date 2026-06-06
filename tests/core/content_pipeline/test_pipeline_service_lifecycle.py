import asyncio
import contextlib
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


@pytest.mark.asyncio
async def test_start_job_dedup_hit_returns_existing_without_scheduling():
    saved = {}
    run_calls = []
    svc = _make_service(saved, run_calls)

    existing_payload = {
        "job_id": "existing1",
        "filename": "a.pdf",
        "subject_id": "a",
        "user_id": "u",
        "status": "extracting",
        "source_s3_key": "k1",
        "page_batch_size": 10,
    }

    async def find_recent_duplicate(user_id, source_s3_key, window_sec):
        return existing_payload

    job = await svc.start_job(
        file_path="/tmp/a.pdf",
        subject_id="a",
        prs_threshold=0.5,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="u",
        content_processor_available=True,
        content_processor_src="",
        page_batch_size=10,
        source_s3_key="k1",
        dedup_window_sec=30,
        find_recent_duplicate=find_recent_duplicate,
    )

    assert job.job_id == "existing1"
    assert run_calls == []
    assert "existing1" not in saved  # no new persist/schedule for the duplicate


@pytest.mark.asyncio
async def test_start_job_no_dedup_creates_and_persists_source():
    saved = {}
    run_calls = []
    svc = _make_service(saved, run_calls)

    async def find_recent_duplicate(user_id, source_s3_key, window_sec):
        return None

    job = await svc.start_job(
        file_path="/tmp/a.pdf",
        subject_id="a",
        prs_threshold=0.5,
        min_confidence=0.6,
        apply_reduction=True,
        user_id="u",
        content_processor_available=True,
        content_processor_src="",
        page_batch_size=10,
        source_s3_key="k1",
        dedup_window_sec=30,
        find_recent_duplicate=find_recent_duplicate,
    )

    assert job.source_s3_key == "k1"
    assert job.prs_threshold == 0.5
    assert job.min_confidence == 0.6
    assert job.apply_reduction is True
    assert job.job_id in saved
    await asyncio.sleep(0)
    assert job.job_id in run_calls


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


def test_build_job_from_payload_preserves_result_and_graph_fields():
    saved = {}
    svc = _make_service(saved, [])
    payload = {
        "job_id": "p9",
        "filename": "a.pdf",
        "subject_id": "a",
        "user_id": "u",
        "status": "failed",
        "progress": 0.6,
        "result": {"graph": {"nodes": [1], "edges": []}, "stats": {"num_nodes": 1}},
        "graph_stats": {"num_nodes": 1, "num_edges": 0, "is_dag": True},
        "partial_graph": {"nodes": [{"id": "c1"}], "edges": []},
        "error_message": "boom",
        "total_chunks": 7,
        "batch_count": 3,
        "failed_batch_count": 1,
        "partial_success": True,
        "concepts_extracted": 5,
        "concepts_after_merge": 4,
        "relations_verified": 2,
    }
    job = svc.build_job_from_payload(payload)
    assert job.result == payload["result"]
    assert job.graph_stats == payload["graph_stats"]
    assert job.partial_graph == payload["partial_graph"]
    assert job.error_message == "boom"
    assert job.total_chunks == 7
    assert job.batch_count == 3
    assert job.failed_batch_count == 1
    assert job.partial_success is True
    assert job.concepts_extracted == 5
    assert job.concepts_after_merge == 4
    assert job.relations_verified == 2


@pytest.mark.asyncio
async def test_reaper_cancels_local_task_before_failing():
    saved = {}
    svc = _make_service(saved, [])

    # Simulate an in-flight background task for "j_inflight".
    inflight = asyncio.create_task(asyncio.sleep(60))
    svc._scheduled_tasks["j_inflight"] = inflight

    async def list_active(**kwargs):
        return [
            {
                "job_id": "j_inflight",
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
    # cancel() was requested on the orphaned local task by the reaper.
    assert inflight.cancelling() > 0
    # The reaper persisted an authoritative FAILED + retryable outcome.
    assert saved["j_inflight"].status is PipelineStatus.FAILED
    assert saved["j_inflight"].error_code == "pipeline_stalled"
    assert saved["j_inflight"].retryable is True

    # Drain the cancelled task and confirm cancellation actually landed,
    # avoiding "task was destroyed" warnings.
    with contextlib.suppress(asyncio.CancelledError):
        await inflight
    assert inflight.cancelled()
    # No *live* (non-done) task remains tracked for the job.
    tracked = svc._scheduled_tasks.get("j_inflight")
    assert tracked is None or tracked.done()


@pytest.mark.asyncio
async def test_cancel_local_task_noop_when_absent():
    saved = {}
    svc = _make_service(saved, [])
    assert svc._cancel_local_task("nope") is False
