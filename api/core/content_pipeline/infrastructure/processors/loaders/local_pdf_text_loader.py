"""Local PDF loader for RAG chunk persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
from loguru import logger

from api.core.content_pipeline.infrastructure.utils.timeit import timeit

from .base import BaseLoader


class LocalPdfTextLoader(BaseLoader):
    """Extract raw page text locally so RAG never depends on an external provider."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    @timeit
    def load(self, file_path: str) -> dict[str, Any]:
        self._validate_file(file_path)

        page_payloads: list[dict[str, Any]] = []
        with fitz.open(file_path) as document:
            for page_index, page in enumerate(document, start=1):
                text = (page.get_text("text") or "").strip()
                page_payloads.append({"page_number": page_index, "text": text})

        rendered_pages = [
            f"## Trang {page['page_number']}\n{page['text']}"
            for page in page_payloads
            if str(page["text"]).strip()
        ]
        text = "\n\n".join(rendered_pages)
        metadata = {
            "file_name": Path(file_path).name,
            "file_path": str(Path(file_path).absolute()),
            "source": "pymupdf",
            "page_count": len(page_payloads),
        }

        logger.info(
            "Loaded PDF locally for chunk persistence",
            file_path=file_path,
            page_count=len(page_payloads),
            text_length=len(text),
        )
        return {
            "text": text,
            "pages": page_payloads,
            "metadata": metadata,
        }
