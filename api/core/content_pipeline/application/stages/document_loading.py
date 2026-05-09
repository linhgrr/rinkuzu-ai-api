"""Document loading stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from .execution import run_blocking_stage

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]
LoadAndChunkFn = Callable[[str, str], list[Any]]


async def load_document_chunks(
    job: PipelineJob,
    *,
    file_path: str,
    load_and_chunk: LoadAndChunkFn,
    persist_job_state: PersistJobStateFn,
) -> list[Any]:
    """Load and chunk a source document while persisting job progress."""
    await persist_job_state(job, PipelineStatus.LOADING, "Loading PDF...", 0.05)

    chunks: list[Any] = await run_blocking_stage(
        load_and_chunk,
        file_path,
        job.subject_id,
        stage_name="document_loading",
    )
    job.total_chunks = len(chunks)

    await persist_job_state(job, PipelineStatus.LOADING, "Loading PDF...", 0.10)
    return chunks
