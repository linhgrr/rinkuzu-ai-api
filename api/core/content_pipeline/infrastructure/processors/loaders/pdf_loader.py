"""PDF document loader using Landing AI."""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from api.config import get_settings

try:
    from agentic_doc.parse import parse

    AGENTIC_DOC_AVAILABLE = True
except ImportError:
    AGENTIC_DOC_AVAILABLE = False

from api.core.content_pipeline.infrastructure.utils import timeit

from .base import BaseLoader


class PDFLoader(BaseLoader):
    """Loader for PDF documents using Landing AI."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize PDFLoader with Landing AI API key.

        Args:
            api_key: Landing AI API key. If not provided, will try to load from
                    unified backend settings.

        Raises:
            ValueError: If API key is not provided and not in environment
            ImportError: If agentic-doc is not installed
        """
        if not AGENTIC_DOC_AVAILABLE:
            raise ImportError(
                "agentic-doc is required for PDF loading with Landing AI. "
                "Install it with: pip install agentic-doc"
            )

        settings = get_settings()
        self.api_key = api_key or settings.vision_agent_api_key
        if not self.api_key:
            raise ValueError(
                "Landing AI API key not provided. "
                "Please set it in backend settings or pass it as parameter."
            )

        os.environ["VISION_AGENT_API_KEY"] = self.api_key

        logger.info("PDFLoader initialized with Landing AI API")

    def supports(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

    @timeit
    def load(self, file_path: str) -> dict[str, Any]:
        """
        Load PDF document using Landing AI and convert to Markdown format.

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with:
                - text: Full markdown content
                - markdown: Full markdown content (same as text)
                - chunks: List of content chunks from Landing AI
                - metadata: Document metadata (filename, page count, etc.)
                - structured_data: Structured chunks from Landing AI
        """
        self._validate_file(file_path)

        try:
            logger.info(f"Starting Landing AI parsing for: {file_path}")

            # Parse PDF using Landing AI
            results = parse(file_path)

        except Exception as e:
            logger.error(f"Error loading PDF with Landing AI: {e!s}", exc_info=True)
            raise

        if not results:
            raise ValueError(f"Landing AI returned no results for: {file_path}")

        result = results[0]

        markdown_content = result.markdown or ""

        chunks = result.chunks if hasattr(result, "chunks") else []

        # Prepare metadata
        metadata = {
            "file_name": Path(file_path).name,
            "file_path": str(Path(file_path).absolute()),
            "source": "landing_ai",
            "num_chunks": len(chunks) if chunks else 0,
        }

        logger.info(
            "Successfully parsed PDF",
            file_path=file_path,
            num_chunks=len(chunks) if chunks else 0,
            markdown_length=len(markdown_content),
        )

        return {
            "text": markdown_content,
            "markdown": markdown_content,
            "chunks": chunks,
            "metadata": metadata,
            "structured_data": chunks,
        }
