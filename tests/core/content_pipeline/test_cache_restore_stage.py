import asyncio

from api.core.content_pipeline.application.stages import cache_restore as cache_restore_stage
from api.core.content_pipeline.application.stages.cache_restore import (
    try_restore_completed_job_from_mongo,
    try_restore_completed_job_from_s3,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


async def _load_completed_job(job_id: str):
    return {
        "job_id": job_id,
        "filename": "lesson.pdf",
        "subject_id": "algebra",
        "status": PipelineStatus.COMPLETED.value,
        "result": {
            "concept_map": {"c1": 0},
            "prereq_edges": [{"source": "c1", "target": "c1"}],
            "stats": {"relations_verified": 1},
        },
        "total_chunks": 4,
        "concepts_extracted": 2,
        "concepts_after_merge": 1,
        "relations_verified": 1,
        "graph_stats": {"num_nodes": 1},
        "completed_at": 123.0,
    }


def test_try_restore_completed_job_from_mongo_populates_job_state():
    job = PipelineJob(job_id="job-1", filename="upload.pdf", subject_id="math")

    restored = asyncio.run(
        try_restore_completed_job_from_mongo(
            job,
            load_job=_load_completed_job,
            populate_metrics=lambda _restored_job: None,
        )
    )

    assert restored is True
    assert job.status == PipelineStatus.COMPLETED
    assert job.filename == "lesson.pdf"
    assert job.subject_id == "algebra"
    assert job.total_chunks == 4
    assert job.progress == 1.0
    assert job.current_step == "Loaded from MongoDB"


async def _load_missing_job(_: str):
    return None


def test_try_restore_completed_job_from_mongo_returns_false_for_miss():
    job = PipelineJob(job_id="job-2", filename="upload.pdf", subject_id="math")

    restored = asyncio.run(
        try_restore_completed_job_from_mongo(
            job,
            load_job=_load_missing_job,
            populate_metrics=lambda _restored_job: None,
        )
    )

    assert restored is False
    assert job.status == PipelineStatus.PENDING


def test_try_restore_completed_job_from_s3_hashes_and_reads_via_blocking_stage(monkeypatch):
    job = PipelineJob(job_id="job-3", filename="upload.pdf", subject_id="math")
    stage_names: list[str] = []

    class _Body:
        def read(self) -> bytes:
            return (
                b'{"concept_map":{"c1":0},"prereq_edges":[],"stats":{"num_nodes":1,"num_edges":0}}'
            )

    class _S3Client:
        def get_object(self, **kwargs):
            assert kwargs["Bucket"] == "bucket-1"
            assert kwargs["Key"] == "cache/hash-123.json"
            return {"Body": _Body()}

    async def fake_run_blocking_stage(func, *args, stage_name, timeout_sec=None, **kwargs):
        assert timeout_sec in {None, 10.0}
        stage_names.append(stage_name)
        return func(*args, **kwargs)

    saved_steps: list[str] = []

    async def save_job(_job):
        saved_steps.append(_job.current_step)
        return True

    monkeypatch.setattr(cache_restore_stage, "run_blocking_stage", fake_run_blocking_stage)

    cache_key = asyncio.run(
        try_restore_completed_job_from_s3(
            job,
            file_path="/tmp/upload.pdf",  # noqa: S108
            s3_client=_S3Client(),
            bucket_name="bucket-1",
            hash_file=lambda _path: "hash-123",
            save_job=save_job,
            populate_metrics=lambda _job: None,
        )
    )

    assert cache_key == "cache/hash-123.json"
    assert stage_names == ["s3_cache_hash", "s3_cache_restore", "s3_cache_body_read"]
    assert saved_steps == ["Kiểm tra cache trên S3...", "Loaded from S3 cache"]
    assert job.status == PipelineStatus.COMPLETED
    assert job.current_step == "Loaded from S3 cache"
