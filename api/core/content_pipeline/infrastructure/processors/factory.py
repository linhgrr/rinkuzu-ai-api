"""Factory for creating file loaders."""


from langchain_core.documents import Document
from loguru import logger

from api.core.content_pipeline.infrastructure.utils import get_file_type

from .errors import LoaderNotFoundError
from .loaders.base import BaseLoader


class FileLoaderFactory:
    """Factory for creating appropriate loaders for files."""

    _loaders: dict[str, type[BaseLoader]] = {}

    @classmethod
    def register(cls, file_type: str, loader_class: type[BaseLoader]):
        """Register a loader for a file type."""
        cls._loaders[file_type] = loader_class
        logger.info(f"Registered loader for {file_type}")

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
            raise LoaderNotFoundError(
                f"No loader found for file type: {file_type}"
            )

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

            return chunks

        except Exception as e:
            logger.error(f"Error loading and chunking file: {e}")
            raise

    @classmethod
    def _get_chunker(cls, file_type: str):
        """Get appropriate chunker for file type."""
        from .chunkers import TextChunker

        # hiện tại chỉ có TextChunker
        # TODO: Thêm các chunker đặc biệt cho slide, video nếu cần
        chunker_map = {
            "pdf": TextChunker(),
            "text": TextChunker(),
        }

        return chunker_map.get(file_type, TextChunker())


def _register_default_loaders():
    """Register default loaders.

    Prefers VisionPDFLoader (S3 + LLM OCR) over Landing AI PDFLoader.
    Falls back to PDFLoader if VisionPDFLoader dependencies are missing.
    """
    # Try VisionPDFLoader first (preferred)
    try:
        from .loaders.vision_pdf_loader import VisionPDFLoader
        FileLoaderFactory.register("pdf", VisionPDFLoader)
        logger.info("Registered VisionPDFLoader for 'pdf' type (S3 + LLM OCR)")
        return  # Done — no need for fallback
    except (ImportError, ValueError) as e:
        logger.warning(f"VisionPDFLoader not available: {e}")

    # Fallback to Landing AI PDFLoader
    try:
        from .loaders.pdf_loader import PDFLoader
        FileLoaderFactory.register("pdf", PDFLoader)
        logger.info("Registered PDFLoader for 'pdf' type (Landing AI fallback)")
    except ImportError as e:
        logger.warning(f"PDFLoader not available: {e}")


_register_default_loaders()
