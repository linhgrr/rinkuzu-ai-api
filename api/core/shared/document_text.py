"""Shared document-text extraction primitives for OCR-backed LLM pipelines."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Protocol, cast

import httpx
from loguru import logger

from api.config import get_settings
from api.core.shared import mongo_store
from api.core.shared.persistence import load_document_ocr_record, save_document_ocr_record


class DocumentTextConfigurationError(RuntimeError):
    """Raised when OCR document-text extraction is not fully configured."""


@dataclass(frozen=True)
class DocumentPageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocumentText:
    text: str
    pages: list[DocumentPageText]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OCRApiConfig:
    endpoint: str
    api_key: str
    model: str
    timeout_sec: float = 120.0


ExtractDocumentTextFn = Callable[[], Awaitable["ExtractedDocumentText"]]
ResolveFileSizeBytesFn = Callable[[], Awaitable[int | None]]


class DocumentTextExtractor(Protocol):
    """Provider boundary for converting a PDF into page-level text."""

    def extract_file(self, file_path: str) -> ExtractedDocumentText:
        raise NotImplementedError

    def extract_bytes(
        self,
        pdf_bytes: bytes,
        *,
        filename: str | None = None,
    ) -> ExtractedDocumentText:
        raise NotImplementedError


class LandingAIDocumentTextExtractor:
    """LandingAI ADE Parse-backed extractor for scanned and image-based PDFs."""

    def __init__(
        self,
        *,
        config: OCRApiConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config or build_ocr_api_config()
        self._client = client

    def extract_file(self, file_path: str) -> ExtractedDocumentText:
        path = Path(file_path)
        self._validate_file(path)
        return self.extract_bytes(path.read_bytes(), filename=path.name)

    def extract_bytes(
        self,
        pdf_bytes: bytes,
        *,
        filename: str | None = None,
    ) -> ExtractedDocumentText:
        if not pdf_bytes:
            raise ValueError("PDF bytes are empty.")

        file_name = filename or "document.pdf"
        files = {"document": (file_name, pdf_bytes, "application/pdf")}
        data = {
            "model": self.config.model,
            "split": "page",
        }

        logger.info(
            "[DocumentText] OCR request start provider=landingai file={} size_bytes={} model={}",
            file_name,
            len(pdf_bytes),
            self.config.model,
        )

        response: httpx.Response
        if self._client is None:
            with httpx.Client(timeout=self.config.timeout_sec) as client:
                response = client.post(
                    self.config.endpoint,
                    headers=_ocr_headers(self.config.api_key),
                    data=data,
                    files=files,
                )
        else:
            response = self._client.post(
                self.config.endpoint,
                headers=_ocr_headers(self.config.api_key),
                data=data,
                files=files,
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[DocumentText] OCR request failed status={} file={} body={}",
                exc.response.status_code,
                file_name,
                exc.response.text[:500],
            )
            raise RuntimeError("OCR API request failed") from exc

        extracted = _landing_ai_to_extracted_text(response.json(), file_name=file_name)
        logger.info(
            "[DocumentText] OCR request done file={} pages={} chars={}",
            file_name,
            extracted.metadata.get("page_count", 0),
            len(extracted.text),
        )
        return extracted

    @staticmethod
    def _validate_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if not path.stat().st_size:
            raise ValueError(f"File is empty: {path}")


def extract_document_text_from_file(file_path: str) -> ExtractedDocumentText:
    return build_document_text_extractor().extract_file(file_path)


def extract_document_text_from_bytes(
    pdf_bytes: bytes,
    *,
    filename: str | None = None,
) -> ExtractedDocumentText:
    return build_document_text_extractor().extract_bytes(pdf_bytes, filename=filename)


def build_page_batches(page_count: int, batch_size: int) -> list[tuple[int, int]]:
    """Return 1-indexed inclusive page windows."""
    if page_count <= 0:
        return []
    normalized_batch_size = max(1, batch_size)
    return [
        (start_page, min(start_page + normalized_batch_size - 1, page_count))
        for start_page in range(1, page_count + 1, normalized_batch_size)
    ]


def build_text_batches(
    pages: list[DocumentPageText],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build page-window text batches consumed by LLM extraction steps."""
    batches: list[dict[str, Any]] = []
    for batch_index, (start_page, end_page) in enumerate(
        build_page_batches(len(pages), batch_size)
    ):
        window = pages[start_page - 1 : end_page]
        batch_text = "\n\n".join(
            f"## Trang {page.page_number}\n{page.text}".strip()
            for page in window
            if page.text.strip()
        ).strip()
        batches.append(
            {
                "batch_index": batch_index,
                "page_start": start_page,
                "page_end": end_page,
                "text": batch_text,
                "char_count": len(batch_text),
            }
        )
    return batches


def calculate_pdf_bytes_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def extracted_document_text_to_content_payload(
    extracted: ExtractedDocumentText,
    *,
    file_hash: str | None = None,
    ocr_cache_hit: bool | None = None,
) -> dict[str, Any]:
    metadata = dict(extracted.metadata or {})
    if file_hash is not None:
        metadata["file_hash"] = file_hash
    if ocr_cache_hit is not None:
        metadata["ocr_cache_hit"] = ocr_cache_hit
    metadata.setdefault("page_count", len(extracted.pages))
    return {
        "text": extracted.text,
        "pages": [{"page_number": page.page_number, "text": page.text} for page in extracted.pages],
        "metadata": metadata,
    }


def ocr_record_to_extracted_document_text(record: dict[str, Any]) -> ExtractedDocumentText:
    metadata = dict(record.get("metadata") or {})
    metadata["file_hash"] = record["file_hash"]
    metadata["ocr_cache_hit"] = True
    metadata.setdefault(
        "page_count", int(record.get("page_count") or len(record.get("pages") or []))
    )
    pages = [
        DocumentPageText(
            page_number=int(page.get("page_number") or 0),
            text=str(page.get("text") or ""),
        )
        for page in record.get("pages") or []
    ]
    return ExtractedDocumentText(
        text=str(record.get("text") or ""),
        pages=pages,
        metadata=metadata,
    )


async def load_or_extract_document_text_cached(
    *,
    file_hash: str,
    file_name: str,
    extract_document_text: ExtractDocumentTextFn,
    file_size_bytes: int | None = None,
    resolve_file_size_bytes: ResolveFileSizeBytesFn | None = None,
) -> ExtractedDocumentText:
    if mongo_store.is_available():
        cached_record = await load_document_ocr_record(file_hash)
        if cached_record is not None:
            return ocr_record_to_extracted_document_text(cached_record)

    document_text = await extract_document_text()
    resolved_file_size_bytes = file_size_bytes
    if resolved_file_size_bytes is None and resolve_file_size_bytes is not None:
        resolved_file_size_bytes = await resolve_file_size_bytes()

    if mongo_store.is_available():
        await save_document_ocr_record(
            file_hash=file_hash,
            file_name=file_name,
            file_size_bytes=resolved_file_size_bytes,
            content=extracted_document_text_to_content_payload(document_text),
        )
    return document_text


def build_document_text_extractor(settings: object | None = None) -> DocumentTextExtractor:
    return LandingAIDocumentTextExtractor(config=build_ocr_api_config(settings))


def build_ocr_api_config(settings: object | None = None) -> OCRApiConfig:
    configured_settings = settings or get_settings()
    endpoint = str(getattr(configured_settings, "ocr_base_url", "") or "").strip()
    api_key = str(getattr(configured_settings, "ocr_api_key", "") or "").strip()
    model = str(getattr(configured_settings, "ocr_model", "") or "").strip()
    timeout_sec = float(getattr(configured_settings, "ocr_timeout_sec", 120))

    if not endpoint:
        raise DocumentTextConfigurationError("OCR API endpoint is not set. Configure OCR_BASE_URL.")
    if not api_key:
        raise DocumentTextConfigurationError("OCR API key is not set. Configure OCR_API_KEY.")
    if not model:
        raise DocumentTextConfigurationError("OCR model is not set. Configure OCR_MODEL.")

    return OCRApiConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        timeout_sec=max(1.0, timeout_sec),
    )


def _ocr_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


_ANCHOR_RE = re.compile(r"<a\s+id=['\"][^'\"]+['\"]></a>", flags=re.IGNORECASE)
_BLANKS_RE = re.compile(r"\n{3,}")


def _landing_ai_to_extracted_text(
    payload: dict[str, Any],
    *,
    file_name: str,
) -> ExtractedDocumentText:
    metadata = cast("dict[str, Any]", payload.get("metadata") or {})
    splits_raw = cast("list[dict[str, Any]]", payload.get("splits") or [])
    page_payloads: list[DocumentPageText] = []

    for split in splits_raw:
        pages = cast("list[int]", split.get("pages") or [])
        if not pages:
            continue
        page_number = int(pages[0])
        markdown = _normalize_landing_ai_markdown(str(split.get("markdown") or "")).strip()
        page_payloads.append(DocumentPageText(page_number=page_number, text=markdown))

    page_payloads.sort(key=lambda item: item.page_number)

    if not page_payloads:
        markdown = _normalize_landing_ai_markdown(str(payload.get("markdown") or "")).strip()
        page_payloads = [DocumentPageText(page_number=1, text=markdown)] if markdown else []

    rendered_pages = [
        f"## Trang {page.page_number}\n{page.text}" for page in page_payloads if page.text.strip()
    ]

    return ExtractedDocumentText(
        text="\n\n".join(rendered_pages).strip(),
        pages=page_payloads,
        metadata={
            "file_name": file_name,
            "file_path": None,
            "source": "ocr_api",
            "provider": "landingai",
            "model": metadata.get("version") or payload.get("model"),
            "page_count": int(metadata.get("page_count") or len(page_payloads)),
            "credit_usage": metadata.get("credit_usage"),
            "job_id": metadata.get("job_id"),
            "failed_pages": metadata.get("failed_pages") or [],
        },
    )


def _normalize_landing_ai_markdown(markdown: str) -> str:
    cleaned = _ANCHOR_RE.sub("", markdown)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _BLANKS_RE.sub("\n\n", cleaned)
    return cleaned.strip()
