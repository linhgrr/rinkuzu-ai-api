"""PDF loader using Vision LLM endpoint for OCR text extraction.

Splits PDF into individual pages, uploads each to S3, then uses
an OpenAI-compatible chat completions endpoint with image_url to
perform OCR on each page in parallel.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from pathlib import Path
import time
from typing import Any
import uuid

import boto3
from botocore.client import Config
import fitz  # pymupdf
from loguru import logger
import requests

from api.config import get_settings
from api.core.shared.llm import build_chat_completions_url, extract_llm_text

from .base import BaseLoader

# ── OCR prompt ─────────────────────────────────────────────────────────
OCR_PROMPT = (
    "You are an expert OCR and document formatting AI. "
    "I am providing a page from a scanned/digital document.\n\n"
    "Your task is to extract ALL the text from this page and "
    "perfectly reconstruct the layout, formatting, and structure.\n\n"
    "Rules:\n"
    "- Output plain text, preserving paragraphs and line breaks.\n"
    "- Preserve headings, bullet points, numbered lists.\n"
    "- Preserve tables as markdown tables.\n"
    "- Preserve bold (**bold**) and italics (*italics*).\n"
    "- Preserve mathematical formulas using LaTeX notation: "
    "inline $...$ and display $$...$$ (e.g., $x^2 + 1$, $$\\Delta = b^2 - 4ac$$).\n"
    "- For Vietnamese text with diacritics, ensure accurate accent reproduction "
    "(e.g., ắ, ề, ổ, ứ, ỹ).\n"
    "- Do NOT add any commentary or explanation.\n"
    "- If the page is blank or has no readable text, output: [BLANK PAGE]\n"
    "- Return ONLY the extracted text content."
)


def _get_s3_client():
    """Create boto3 S3 client from unified settings."""
    settings = get_settings()
    endpoint_url = settings.s3_endpoint_url
    access_key = settings.s3_access_key_id
    secret_key = settings.s3_secret_access_key

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError(
            "S3 credentials not configured. "
            "Set S3 endpoint and credentials in backend settings."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={"addressing_style": "path"}),
    )


def _split_pdf_to_pages(file_path: str) -> list[bytes]:
    """Split a PDF file into individual single-page PDF byte buffers."""
    doc = fitz.open(file_path)
    pages: list[bytes] = []

    for page_num in range(len(doc)):
        # Create a new single-page PDF
        single_page_doc = fitz.open()
        single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        page_bytes = single_page_doc.tobytes()
        single_page_doc.close()
        pages.append(page_bytes)

    doc.close()
    logger.info(f"[VisionPDFLoader] Split PDF into {len(pages)} pages")
    return pages


def _upload_page_to_s3(
    s3_client,
    bucket_name: str,
    endpoint_url: str,
    page_bytes: bytes,
    page_num: int,
    job_id: str,
) -> str:
    """Upload a single page PDF to S3 and return the public URL."""
    timestamp = int(time.time() * 1000)
    key = f"pdfs/{job_id}/{timestamp}-page-{page_num}.pdf"

    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=page_bytes,
        ContentType="application/pdf",
        ContentDisposition="inline",
    )

    base_url = endpoint_url.rstrip("/")
    url = f"{base_url}/{bucket_name}/{key}"
    logger.debug(f"[VisionPDFLoader] Uploaded page {page_num} → {url}")
    return url


def _ocr_page_via_llm(
    page_url: str,
    page_num: int,
    base_url: str,
    model: str,
    api_key: str,
    request_timeout_sec: float,
    max_retries: int = 2,
) -> tuple[int, str]:
    """Call LLM chat completions endpoint to OCR a single page.

    Returns:
        Tuple of (page_num, extracted_text)
    """
    url = build_chat_completions_url(base_url)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": page_url},
                    },
                ],
            }
        ],
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract content from OpenAI-compatible response
            content = extract_llm_text(
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not content:
                logger.warning(
                    f"[VisionPDFLoader] Page {page_num}: empty response from LLM"
                )
                return (page_num, "")

            logger.info(
                f"[VisionPDFLoader] Page {page_num} OCR completed "
                f"({len(content)} chars)"
            )
            return (page_num, content)

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"[VisionPDFLoader] Page {page_num} attempt {attempt + 1} "
                    f"failed: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"[VisionPDFLoader] Page {page_num} failed after "
                    f"{max_retries + 1} attempts: {last_error}"
                )

    return (page_num, f"[OCR_ERROR: {last_error}]")


def _cleanup_s3_pages(s3_client, bucket_name: str, job_id: str):
    """Delete temporary page PDFs from S3 (one-by-one for compatibility)."""
    try:
        prefix = f"pdfs/{job_id}/"
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        objects = response.get("Contents", [])
        deleted = 0
        for obj in objects:
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
                deleted += 1
            except Exception:
                pass
        if deleted:
            logger.info(
                f"[VisionPDFLoader] Cleaned up {deleted} temp files "
                f"from S3 prefix {prefix}"
            )
    except Exception as e:
        logger.warning(f"[VisionPDFLoader] Cleanup failed: {e}")


class VisionPDFLoader(BaseLoader):
    """PDF loader using Vision LLM for OCR with parallel page processing."""

    def __init__(
        self,
        concurrency: int | None = None,
        cleanup_s3: bool = True,
    ):
        """
        Args:
            concurrency: Max parallel OCR requests (default: backend settings)
            cleanup_s3: Whether to delete temp S3 page files after extraction
        """
        settings = get_settings()
        self.settings = settings
        self.concurrency = concurrency or settings.pdf_ocr_concurrency
        self.cleanup_s3 = cleanup_s3

        # LLM config
        self.base_url = settings.llm_base_url or "http://localhost:6969"
        self.model = settings.llm_model or "gemini-3.0-pro"
        self.api_key = (
            settings.llm_api_key
            or settings.gemini_api_key
            or settings.google_api_key
            or "sk-41bb5a29c07d4b23ad5e8e54a658ce2b"
        )
        self.request_timeout_sec = settings.vision_pdf_request_timeout_sec
        self.max_retries = settings.llm_max_retries

        # S3 config
        self.bucket_name = settings.s3_bucket_name or ""
        self.s3_endpoint_url = settings.s3_endpoint_url or ""
        self.s3_client = _get_s3_client()

        logger.info(
            f"[VisionPDFLoader] Initialized: "
            f"concurrency={self.concurrency}, model={self.model}, "
            f"base_url={self.base_url}"
        )

    def supports(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

    def load(self, file_path: str) -> dict[str, Any]:
        """
        Load PDF by splitting into pages, uploading to S3,
        and extracting text via LLM vision endpoint in parallel.

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with:
                - text: Full extracted text content
                - markdown: Same as text
                - chunks: List of per-page text chunks
                - metadata: Document metadata
                - structured_data: Per-page structured data
        """
        self._validate_file(file_path)

        job_id = str(uuid.uuid4())[:12]
        filename = Path(file_path).name

        logger.info(
            f"[VisionPDFLoader] Starting OCR for: {filename} (job: {job_id})"
        )

        try:
            # Step 1: Split PDF into pages
            page_buffers = _split_pdf_to_pages(file_path)
            total_pages = len(page_buffers)

            if total_pages == 0:
                raise ValueError(f"PDF has 0 pages: {file_path}")

            # Step 2: Upload all pages to S3
            logger.info(
                f"[VisionPDFLoader] Uploading {total_pages} pages to S3..."
            )
            page_urls: list[tuple[int, str]] = []
            for i, page_bytes in enumerate(page_buffers):
                url = _upload_page_to_s3(
                    self.s3_client,
                    self.bucket_name,
                    self.s3_endpoint_url,
                    page_bytes,
                    i + 1,
                    job_id,
                )
                page_urls.append((i + 1, url))

            # Step 3: OCR all pages in parallel
            logger.info(
                f"[VisionPDFLoader] Starting parallel OCR "
                f"({self.concurrency} workers)..."
            )
            results: dict[int, str] = {}

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(
                        _ocr_page_via_llm,
                        url,
                        page_num,
                        self.base_url,
                        self.model,
                        self.api_key,
                        self.request_timeout_sec,
                        self.max_retries,
                    ): page_num
                    for page_num, url in page_urls
                }

                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        pnum, text = future.result()
                        results[pnum] = text
                    except Exception as e:
                        logger.error(
                            f"[VisionPDFLoader] Page {page_num} error: {e}"
                        )
                        results[page_num] = f"[OCR_ERROR: {e}]"

            # Step 4: Merge results in page order
            ordered_texts = []
            page_chunks = []
            for page_num in sorted(results.keys()):
                text = results[page_num]
                if text and text.strip() and text.strip() != "[BLANK PAGE]":
                    ordered_texts.append(text)
                    page_chunks.append(
                        {
                            "page": page_num,
                            "text": text,
                        }
                    )

            full_text = "\n\n---\n\n".join(ordered_texts)

            # Step 5: Cleanup S3 temp files
            if self.cleanup_s3:
                _cleanup_s3_pages(self.s3_client, self.bucket_name, job_id)

            # Build metadata
            metadata = {
                "file_name": filename,
                "file_path": str(Path(file_path).absolute()),
                "source": "vision_llm_ocr",
                "total_pages": total_pages,
                "extracted_pages": len(page_chunks),
                "model": self.model,
                "num_chunks": len(page_chunks),
            }

            logger.info(
                f"[VisionPDFLoader] ✅ Completed: {filename} — "
                f"{len(page_chunks)}/{total_pages} pages extracted, "
                f"{len(full_text)} chars total"
            )

            return {
                "text": full_text,
                "markdown": full_text,
                "chunks": page_chunks,
                "metadata": metadata,
                "structured_data": page_chunks,
            }

        except Exception as e:
            # Attempt cleanup even on failure
            if self.cleanup_s3:
                with contextlib.suppress(Exception):
                    _cleanup_s3_pages(
                        self.s3_client, self.bucket_name, job_id
                    )
            logger.error(
                f"[VisionPDFLoader] Error processing {filename}: {e}",
                exc_info=True,
            )
            raise
