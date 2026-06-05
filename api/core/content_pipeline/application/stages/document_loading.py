"""Document loading stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus
from api.core.content_pipeline.infrastructure.processors.factory import chunk_document_content
from api.core.content_pipeline.infrastructure.runtime import calculate_file_hash
from api.core.shared.document_text import (
    extract_document_text_from_file,
    extracted_document_text_to_content_payload,
    load_or_extract_document_text_cached,
)

from .execution import run_blocking_stage

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def _file_size_bytes(file_path: str) -> int:
    return int(Path(file_path).stat().st_size)


async def load_document_chunks(
    job: PipelineJob,
    *,
    file_path: str,
    persist_job_state: PersistJobStateFn,
) -> list[Any]:
    """Load and chunk a source document while persisting job progress."""
    await persist_job_state(
        job, PipelineStatus.LOADING, "Loading PDF...", PipelineProgress.PDF_LOADED
    )

    file_hash = await run_blocking_stage(
        calculate_file_hash,
        file_path,
        stage_name="document_hashing",
    )
    document_text = await load_or_extract_document_text_cached(
        file_hash=file_hash,
        file_name=Path(file_path).name,
        extract_document_text=lambda: run_blocking_stage(
            extract_document_text_from_file,
            file_path,
            stage_name="document_ocr_loading",
        ),
        resolve_file_size_bytes=lambda: run_blocking_stage(
            _file_size_bytes,
            file_path,
            stage_name="document_size_stat",
        ),
    )
    content = extracted_document_text_to_content_payload(
        document_text,
        file_hash=file_hash,
        ocr_cache_hit=bool(document_text.metadata.get("ocr_cache_hit", False)),
    )

    job.total_pages = int((content.get("metadata") or {}).get("page_count") or 0)
    chunks: list[Any] = await run_blocking_stage(
        chunk_document_content,
        content,
        job.subject_id,
        stage_name="document_chunking",
    )
    job.total_chunks = len(chunks)

    await persist_job_state(
        job, PipelineStatus.LOADING, "Loading PDF...", PipelineProgress.PDF_CHUNKED
    )
    return chunks
