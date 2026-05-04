import asyncio

from api.core.content_pipeline.application.stages.document_loading import load_document_chunks
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_load_document_chunks_updates_progress_and_total_chunks():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def load_and_chunk(file_path: str, subject_id: str):
        assert file_path == "fixtures/lesson.pdf"
        assert subject_id == "algebra"
        return ["chunk-1", "chunk-2", "chunk-3"]

    chunks = asyncio.run(
        load_document_chunks(
            job,
            file_path="fixtures/lesson.pdf",
            load_and_chunk=load_and_chunk,
            persist_job_state=persist_job_state,
        )
    )

    assert chunks == ["chunk-1", "chunk-2", "chunk-3"]
    assert job.total_chunks == 3
    assert calls == [
        (PipelineStatus.LOADING, "Loading PDF...", 0.05),
        (PipelineStatus.LOADING, "Loading PDF...", 0.10),
    ]
