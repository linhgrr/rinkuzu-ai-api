"""Lightweight OCR text -> chunk transformation helper."""

from __future__ import annotations

from loguru import logger

from .chunkers import TextChunker


def chunk_document_content(content: dict, doc_id: str) -> list:
    """Split OCR-loaded document content into LangChain Documents."""
    chunker = TextChunker()
    chunks = chunker.chunk(content, doc_id)
    logger.info("Chunked OCR document content", doc_id=doc_id, num_chunks=len(chunks))
    return chunks
