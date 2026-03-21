import asyncio

from api.core.content_pipeline.application.stages.finalization import (
    complete_pipeline_job,
    persist_terminal_failure,
    upload_result_cache,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_complete_pipeline_job_persists_completed_status():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    job.concepts_after_merge = 3
    calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    asyncio.run(
        complete_pipeline_job(
            job,
            persist_job_state=persist_job_state,
        )
    )

    assert job.completed_at is not None
    assert calls == [
        (PipelineStatus.COMPLETED, "Processing complete!", 1.0),
    ]


class _S3ClientStub:
    def __init__(self) -> None:
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)


def test_upload_result_cache_writes_json_payload_when_s3_is_configured():
    s3_client = _S3ClientStub()

    asyncio.run(
        upload_result_cache(
            result={"value": "xin chao"},
            s3_client=s3_client,
            bucket_name="bucket-1",
            cache_key="cache/job-1.json",
        )
    )

    assert len(s3_client.calls) == 1
    assert s3_client.calls[0]["Bucket"] == "bucket-1"
    assert s3_client.calls[0]["Key"] == "cache/job-1.json"
    assert s3_client.calls[0]["ContentType"] == "application/json"
    assert "\"xin chao\"" in s3_client.calls[0]["Body"]


def test_persist_terminal_failure_updates_job_and_saves_once():
    job = PipelineJob(job_id="job-2", filename="lesson.pdf", subject_id="algebra")
    saved_states = []

    async def save_job(job_arg):
        saved_states.append((job_arg.status, job_arg.current_step, job_arg.error_message))
        return True

    asyncio.run(
        persist_terminal_failure(
            job,
            error=RuntimeError("boom"),
            save_job=save_job,
        )
    )

    assert job.status == PipelineStatus.FAILED
    assert job.current_step == "Error: boom"
    assert job.error_message == "boom"
    assert saved_states == [
        (PipelineStatus.FAILED, "Error: boom", "boom"),
    ]
