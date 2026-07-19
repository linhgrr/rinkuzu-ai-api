from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from beanie.odm.operators.update.general import Set
from loguru import logger

from .documents import DocumentChunkDocument

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument
    from pymongo.asynchronous.client_session import AsyncClientSession


async def replace_job_chunks(
    *,
    job_id: str,
    subject_id: str,
    generation: int,
    chunks: list[LangChainDocument],
) -> int:
    """Bulk upsert chunks for one immutable job generation."""
    if not chunks:
        return 0
    now = datetime.now(UTC)
    async with DocumentChunkDocument.bulk_writer(ordered=False) as bulk:
        for idx, chunk in enumerate(chunks):
            chunk_index = int(chunk.metadata.get("chunk_index", idx))
            start_page = int(chunk.metadata.get("start_page", 0) or 0)
            end_page = int(chunk.metadata.get("end_page", 0) or 0)
            doc = DocumentChunkDocument(
                job_id=job_id,
                subject_id=subject_id,
                generation=generation,
                chunk_index=chunk_index,
                text=chunk.page_content,
                start_page=start_page,
                end_page=end_page,
                created_at=now,
            )
            await DocumentChunkDocument.find_one(
                DocumentChunkDocument.job_id == job_id,
                DocumentChunkDocument.generation == generation,
                DocumentChunkDocument.chunk_index == chunk_index,
            ).upsert(
                Set(
                    {
                        DocumentChunkDocument.subject_id: subject_id,
                        DocumentChunkDocument.text: chunk.page_content,
                        DocumentChunkDocument.start_page: start_page,
                        DocumentChunkDocument.end_page: end_page,
                        DocumentChunkDocument.created_at: now,
                    }
                ),
                on_insert=doc,
                bulk_writer=bulk,
            )
        await DocumentChunkDocument.find(
            DocumentChunkDocument.job_id == job_id,
            DocumentChunkDocument.generation == generation,
            {"chunk_index": {"$gte": len(chunks)}},
        ).delete(bulk_writer=bulk)
    logger.info(
        "[DocumentChunkStore] persisted {} chunks job_id={} generation={} subject_id={}",
        len(chunks),
        job_id,
        generation,
        subject_id,
    )
    return len(chunks)


async def delete_chunks_for_job(
    job_id: str,
    *,
    session: AsyncClientSession | None = None,
) -> int:
    result = await DocumentChunkDocument.find(
        DocumentChunkDocument.job_id == job_id,
        session=session,
    ).delete(session=session)
    return 0 if result is None else int(result.deleted_count)
