"""
chunk_persistence.py — Persist document chunks to MongoDB + ChromaDB.

Runs after document_loading, before concept_extraction.
Chunks are stored in both:
  - MongoDB "al_document_chunks" (durability)
  - ChromaDB "document_chunks" (vector retrieval for RAG)
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, List, Optional

from langchain_core.documents import Document as LangChainDocument
from loguru import logger

from ....core.shared import mongo_store
from ...domain.jobs import PipelineJob, PipelineStatus
from ...infrastructure.storage.chunk_chroma_store import ChunkChromaStore


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def persist_document_chunks(
    job: PipelineJob,
    *,
    chunks: List[LangChainDocument],
    chunk_chroma_store: Optional[ChunkChromaStore],
    mongo_collection: Optional[Any],  # AsyncIOMotorCollection
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
        0.11,
    )

    # ── MongoDB ────────────────────────────────────────────────
    if mongo_collection is not None:
        try:
            docs_to_upsert = [
                {
                    "job_id": job.job_id,
                    "subject_id": job.subject_id,
                    "chunk_index": c.metadata.get("chunk_index", i),
                    "text": c.page_content,
                    "start_page": c.metadata.get("start_page", 0),
                    "end_page": c.metadata.get("end_page", 0),
                    "created_at": time.time(),
                }
                for i, c in enumerate(chunks)
            ]
            for doc in docs_to_upsert:
                await mongo_collection.update_one(
                    {"job_id": doc["job_id"], "chunk_index": doc["chunk_index"]},
                    {"$set": doc},
                    upsert=True,
                )
            logger.info(
                f"[persist_chunks] MongoDB: persisted {len(docs_to_upsert)} chunks",
                job_id=job.job_id,
            )
        except Exception as exc:
            logger.warning(
                f"[persist_chunks] MongoDB write failed, continuing pipeline: {exc}",
                job_id=job.job_id,
            )

    # ── ChromaDB ───────────────────────────────────────────────
    if chunk_chroma_store is not None:
        try:
            ids = chunk_chroma_store.add_chunks(
                chunks=chunks,
                job_id=job.job_id,
                subject_id=job.subject_id,
            )
            logger.info(
                f"[persist_chunks] ChromaDB: added {len(ids)} chunks",
                job_id=job.job_id,
            )
        except Exception as exc:
            logger.warning(
                f"[persist_chunks] ChromaDB write failed, continuing pipeline: {exc}",
                job_id=job.job_id,
            )

    await persist_job_state(
        job,
        PipelineStatus.LOADING,
        "Document chunks persisted",
        0.12,
    )

    return len(chunks)
