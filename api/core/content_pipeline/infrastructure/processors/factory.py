"""Lightweight PDF loading + chunking helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .chunkers import TextChunker
from .loaders.local_pdf_text_loader import load_pdf

if TYPE_CHECKING:
    from langchain_core.documents import Document


def load_and_chunk_pdf(file_path: str, doc_id: str) -> list[Document]:
    """Load a PDF file and split its content into LangChain Documents."""
    try:
        content = load_pdf(file_path)
        chunker = TextChunker()
        chunks = chunker.chunk(content, doc_id)
        logger.info(
            "Loaded and chunked PDF",
            file_path=file_path,
            num_chunks=len(chunks),
        )
    except Exception:
        logger.exception("Error loading and chunking PDF")
        raise
    else:
        return chunks
