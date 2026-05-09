"""Factory for creating file loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from api.core.content_pipeline.infrastructure.utils import get_file_type

from .chunkers import TextChunker
from .errors import LoaderNotFoundError
from .loaders.local_pdf_text_loader import LocalPdfTextLoader

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from .loaders.base import BaseLoader

class FileLoaderFactory:
    """Factory for creating appropriate loaders for files."""

    _loaders: ClassVar[dict[str, type[BaseLoader]]] = {}

    @classmethod
    def register(cls, file_type: str, loader_class: type[BaseLoader]):
        """Register a loader for a file type."""
        cls._loaders[file_type] = loader_class
        logger.info("Registered loader for {}", file_type)

    @classmethod
    def get_loader(cls, file_path: str) -> BaseLoader:
        """
        Get appropriate loader for a file.

        Args:
            file_path: Path to file

        Returns:
            Loader instance
        """
        file_type = get_file_type(file_path)

        loader_class = cls._loaders.get(file_type)

        if not loader_class:
            raise LoaderNotFoundError(f"No loader found for file type: {file_type}")

        return loader_class()

    @classmethod
    def load_and_chunk(
        cls,
        file_path: str,
        doc_id: str,
    ) -> list[Document]:
        """
        Load file and chunk its content.

        Args:
            file_path: Path to file
            doc_id: Document ID

        Returns:
            List of Document chunks
        """
        try:
            loader = cls.get_loader(file_path)
            content = loader.load(file_path)
            file_type = get_file_type(file_path)
            chunker = cls._get_chunker(file_type)
            chunks = chunker.chunk(content, doc_id)
            logger.info(
                "Loaded and chunked file",
                file_path=file_path,
                num_chunks=len(chunks),
            )
        except Exception as e:
            logger.exception("Error loading and chunking file")
            raise
        else:
            return chunks

    @classmethod
    def _get_chunker(cls, file_type: str) -> TextChunker:
        """Get appropriate chunker for file type."""
        chunker_map = {
            "pdf": TextChunker(),
            "text": TextChunker(),
        }
        return chunker_map.get(file_type, TextChunker())


def _register_default_loaders():
    """Register default loaders."""
    FileLoaderFactory.register("pdf", LocalPdfTextLoader)
    logger.info("Registered LocalPdfTextLoader for 'pdf' type")


_register_default_loaders()
