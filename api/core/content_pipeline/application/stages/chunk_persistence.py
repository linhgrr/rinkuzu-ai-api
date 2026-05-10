"""
chunk_persistence.py — Persist document chunks to MongoDB + ChromaDB.

Runs after document_loading, before concept_extraction.
Chunks are stored in both:
  - MongoDB "al_document_chunks" (durability)
  - ChromaDB "document_chunks" (vector retrieval for RAG)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pymongo import UpdateOne

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from .execution import run_blocking_stage, safe_run

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument

    from api.core.content_pipeline.infrastructure.storage.chunk_chroma_store import ChunkChromaStore

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def persist_document_chunks(
    job: PipelineJob,
    *,
    chunks: list[LangChainDocument],
    chunk_chroma_store: ChunkChromaStore | None,
    mongo_collection: Any | None,  # AsyncIOMotorCollection
    persist_job_state: PersistJobStateFn,
) -> int:
    """Persist document chunks to MongoDB and ChromaDB.

    Args:
        job: PipelineJob (must have job_id and subject_id).
        chunks: Document chunks from the text chunker.
        chunk_chroma_store: ChunkChromaStore instance (None → skip ChromaDB).
        mongo_collection: Motor collection "al_document_chunks" (None → skip MongoDB).
        persist_job_state: Progress callback.

    Returns:
        Number of chunks persisted (0 if skipped).
    """
    if not chunks:
        logger.info("[persist_chunks] No chunks to persist")
        return 0

    await persist_job_state(
        job,
        PipelineStatus.LOADING,
        "Persisting document chunks...",
        PipelineProgress.CHUNKS_PERSISTING,
    )

    # ── MongoDB ────────────────────────────────────────────────
    if mongo_collection is not None:
        async def _persist_mongo():
            docs_to_upsert = []
            operations = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "job_id": job.job_id,
                    "subject_id": job.subject_id,
                    "chunk_index": chunk.metadata.get("chunk_index", i),
                    "text": chunk.page_content,
                    "start_page": chunk.metadata.get("start_page", 0),
                    "end_page": chunk.metadata.get("end_page", 0),
                    "created_at": time.time(),
                }
                docs_to_upsert.append(doc)
                operations.append(
                    UpdateOne(
                        {"job_id": doc["job_id"], "chunk_index": doc["chunk_index"]},
                        {"$set": doc},
                        upsert=True,
                    )
                )
            if operations:
                await mongo_collection.bulk_write(operations, ordered=False)
            logger.info(
                "[persist_chunks] MongoDB: persisted {} chunks",
                len(docs_to_upsert),
                job_id=job.job_id,
            )

        await safe_run(
            _persist_mongo,
            fail_message="persist_chunks MongoDB write failed, continuing pipeline",
        )

    # ── ChromaDB ───────────────────────────────────────────────
    if chunk_chroma_store is not None:
        async def _persist_chroma():
            ids = await run_blocking_stage(
                chunk_chroma_store.add_chunks,
                chunks=chunks,
                job_id=job.job_id,
                subject_id=job.subject_id,
                stage_name="chroma_add_chunks",
            )
            logger.info(
                "[persist_chunks] ChromaDB: added {} chunks",
                len(ids),
                job_id=job.job_id,
            )

        await safe_run(
            _persist_chroma,
            fail_message="persist_chunks ChromaDB write failed, continuing pipeline",
        )

    await persist_job_state(
        job,
        PipelineStatus.LOADING,
        "Document chunks persisted",
        PipelineProgress.CHUNKS_PERSISTED,
    )

    return len(chunks)
