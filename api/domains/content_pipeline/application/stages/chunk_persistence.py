"""
chunk_persistence.py — Persist document chunks to MongoDB + ChromaDB.

Runs after document_loading, before concept_extraction.
Chunks are stored in both:
  - MongoDB "al_document_chunks" (durability)
  - ChromaDB "document_chunks" (vector retrieval for RAG)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus
from api.shared.persistence import replace_job_chunks

from .execution import run_blocking_stage

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument

    from api.domains.content_pipeline.infrastructure.storage.chunk_chroma_store import (
        ChunkChromaStore,
    )

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def persist_document_chunks(
    job: PipelineJob,
    *,
    chunks: list[LangChainDocument],
    chunk_chroma_store: ChunkChromaStore | None,
    persist_job_state: PersistJobStateFn,
) -> int:
    """Persist chunks to Mongo then Chroma. Storage failures propagate."""
    if not chunks:
        logger.info("[persist_chunks] No chunks to persist")
        return 0

    await persist_job_state(
        job,
        PipelineStatus.LOADING,
        "Persisting document chunks...",
        PipelineProgress.CHUNKS_PERSISTING,
    )

    persisted = await replace_job_chunks(
        job_id=job.job_id,
        subject_id=job.subject_id,
        generation=job.retry_count,
        chunks=chunks,
    )
    logger.info(
        "[persist_chunks] MongoDB: persisted {} chunks",
        persisted,
        job_id=job.job_id,
    )

    if chunk_chroma_store is not None:
        ids = await run_blocking_stage(
            chunk_chroma_store.replace_chunks,
            chunks=chunks,
            job_id=job.job_id,
            subject_id=job.subject_id,
            generation=job.retry_count,
            stage_name="chroma_replace_chunks",
        )
        logger.info(
            "[persist_chunks] ChromaDB: added {} chunks",
            len(ids),
            job_id=job.job_id,
        )

    await persist_job_state(
        job,
        PipelineStatus.LOADING,
        "Document chunks persisted",
        PipelineProgress.CHUNKS_PERSISTED,
    )

    return len(chunks)
