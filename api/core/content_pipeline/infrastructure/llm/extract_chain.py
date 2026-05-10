"""OpenAI Files/Responses extraction chain for content-pipeline concept work."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import time
from typing import Any

import fitz
from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.application.stages.execution import run_process_stage
from api.core.content_pipeline.infrastructure.prompts import (
    EVIDENCE_VERIFICATION_PROMPT,
    EXTRACTION_PROMPT,
)
from api.core.content_pipeline.infrastructure.utils.timeit import atimeit

from .openai_responses import (
    FileReferenceError,
    OpenAIResponsesClient,
    PayloadTooLargeError,
    StructuredExtractionClient,
    require_parsed_output,
    response_usage_summary,
)
from .schemas import (
    ConceptExtraction,
    ConceptExtractionPayload,
    EvidenceVerification,
    materialize_concept_extraction,
)

_COMPRESSION_PROFILES: tuple[tuple[int, int], ...] = (
    (144, 75),
    (110, 60),
    (96, 45),
    (72, 35),
)

_READ_PAGE_COUNT_PROCESS_TARGET = (
    "api.core.content_pipeline.infrastructure.llm.extract_chain:_read_page_count_from_path"
)
_RENDER_BATCHES_PROCESS_TARGET = (
    "api.core.content_pipeline.infrastructure.llm.extract_chain:_render_batched_pdfs_from_path"
)
_SPLIT_BATCH_PROCESS_TARGET = (
    "api.core.content_pipeline.infrastructure.llm.extract_chain:_split_rendered_batch_from_path"
)


class ProviderUploadTooLargeError(RuntimeError):
    """Raised when the provider rejects a PDF batch request body size."""


def build_page_batches(page_count: int, batch_size: int) -> list[tuple[int, int]]:
    """Return 1-indexed inclusive page windows."""
    if page_count <= 0:
        return []
    normalized_batch_size = max(1, batch_size)
    return [
        (start_page, min(start_page + normalized_batch_size - 1, page_count))
        for start_page in range(1, page_count + 1, normalized_batch_size)
    ]


def _format_pages(start_page: int, end_page: int) -> str:
    return f"{start_page}-{end_page}"


def _format_usage(usage: dict[str, int]) -> str:
    if not usage:
        return "-"
    return (
        f"in={usage.get('input_tokens', 0)} "
        f"out={usage.get('output_tokens', 0)} "
        f"total={usage.get('total_tokens', 0)}"
    )


def _read_page_count_from_path(file_path: str) -> int:
    with fitz.open(file_path) as document:
        return int(document.page_count)


def _build_rendered_batch_payload(
    *,
    batch_index: int,
    start_page: int,
    end_page: int,
    pdf_bytes: bytes,
    compression_applied: bool = False,
    compression_profile: str | None = None,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "page_start": start_page,
        "page_end": end_page,
        "pdf_bytes": pdf_bytes,
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "size_bytes": len(pdf_bytes),
        "compression_applied": compression_applied,
        "compression_profile": compression_profile,
    }


def _extract_pdf_bytes_from_document(
    document: fitz.Document,
    start_page: int,
    end_page: int,
) -> bytes:
    with fitz.open() as sub_document:
        sub_document.insert_pdf(document, from_page=start_page - 1, to_page=end_page - 1)
        return sub_document.tobytes(garbage=4, deflate=True)  # type: ignore[no-any-return]


def _extract_compressed_pdf_bytes_from_document(
    document: fitz.Document,
    start_page: int,
    end_page: int,
    *,
    dpi: int,
    jpg_quality: int,
) -> bytes:
    with fitz.open() as compressed_document:
        for page_number in range(start_page - 1, end_page):
            source_page = document[page_number]
            pixmap = source_page.get_pixmap(dpi=dpi, alpha=False)
            image_bytes = pixmap.tobytes("jpeg", jpg_quality=jpg_quality)
            target_page = compressed_document.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            target_page.insert_image(target_page.rect, stream=image_bytes)
        return compressed_document.tobytes(garbage=4, deflate=True)  # type: ignore[no-any-return]


def _compress_rendered_batch_for_document(
    document: fitz.Document,
    *,
    batch_index: int,
    start_page: int,
    end_page: int,
    original_size_bytes: int,
    max_bytes: int,
) -> dict[str, Any] | None:
    best_batch: dict[str, Any] | None = None
    best_size = original_size_bytes

    for dpi, jpg_quality in _COMPRESSION_PROFILES:
        compressed_bytes = _extract_compressed_pdf_bytes_from_document(
            document,
            start_page,
            end_page,
            dpi=dpi,
            jpg_quality=jpg_quality,
        )
        compressed_batch = _build_rendered_batch_payload(
            batch_index=batch_index,
            start_page=start_page,
            end_page=end_page,
            pdf_bytes=compressed_bytes,
            compression_applied=True,
            compression_profile=f"{dpi}dpi-q{jpg_quality}",
        )
        if compressed_batch["size_bytes"] >= best_size:
            continue

        best_batch = compressed_batch
        best_size = compressed_batch["size_bytes"]
        logger.debug(
            "compressed provider batch candidate batch={} pages={} profile={} original_size_bytes={} compressed_size_bytes={} meets_target={}",
            batch_index,
            _format_pages(start_page, end_page),
            f"{dpi}dpi-q{jpg_quality}",
            original_size_bytes,
            best_size,
            best_size <= max_bytes,
        )
        if best_size <= max_bytes:
            break

    return best_batch


def _render_batched_pdfs_for_document(
    document: fitz.Document,
    *,
    batch_index: int,
    start_page: int,
    end_page: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    pdf_bytes = _extract_pdf_bytes_from_document(document, start_page, end_page)
    rendered_batch = _build_rendered_batch_payload(
        batch_index=batch_index,
        start_page=start_page,
        end_page=end_page,
        pdf_bytes=pdf_bytes,
    )

    if rendered_batch["size_bytes"] > max_bytes:
        compressed_batch = _compress_rendered_batch_for_document(
            document,
            batch_index=batch_index,
            start_page=start_page,
            end_page=end_page,
            original_size_bytes=rendered_batch["size_bytes"],
            max_bytes=max_bytes,
        )
        if compressed_batch is not None:
            rendered_batch = compressed_batch

    if rendered_batch["size_bytes"] <= max_bytes or start_page == end_page:
        logger.debug(
            "prepared provider batch batch={} pages={} size_bytes={} compressed={} profile={}",
            batch_index,
            _format_pages(start_page, end_page),
            rendered_batch["size_bytes"],
            rendered_batch.get("compression_applied", False),
            rendered_batch.get("compression_profile") or "-",
        )
        return [{**rendered_batch}]

    midpoint = (start_page + end_page) // 2
    logger.warning(
        "provider batch over byte limit batch={} pages={} size_bytes={} max_bytes={} split_at={} compressed={}",
        batch_index,
        _format_pages(start_page, end_page),
        rendered_batch["size_bytes"],
        max_bytes,
        midpoint,
        rendered_batch.get("compression_applied", False),
    )
    return [
        *_render_batched_pdfs_for_document(
            document,
            batch_index=batch_index,
            start_page=start_page,
            end_page=midpoint,
            max_bytes=max_bytes,
        ),
        *_render_batched_pdfs_for_document(
            document,
            batch_index=batch_index,
            start_page=midpoint + 1,
            end_page=end_page,
            max_bytes=max_bytes,
        ),
    ]


def _render_batched_pdfs_from_path(
    file_path: str,
    batch_index: int,
    start_page: int,
    end_page: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    with fitz.open(file_path) as document:
        return _render_batched_pdfs_for_document(
            document,
            batch_index=batch_index,
            start_page=start_page,
            end_page=end_page,
            max_bytes=max_bytes,
        )


def _split_rendered_batch_from_path(
    file_path: str,
    batch: dict[str, Any],
    max_bytes: int,
) -> list[dict[str, Any]]:
    start_page = int(batch["page_start"])
    end_page = int(batch["page_end"])
    if start_page >= end_page:
        return []
    midpoint = (start_page + end_page) // 2
    logger.warning(
        "provider batch split after upstream rejection batch={} pages={} size_bytes={} split_at={}",
        batch["batch_index"],
        _format_pages(start_page, end_page),
        batch["size_bytes"],
        midpoint,
    )
    with fitz.open(file_path) as document:
        return [
            *_render_batched_pdfs_for_document(
                document,
                batch_index=int(batch["batch_index"]),
                start_page=start_page,
                end_page=midpoint,
                max_bytes=max_bytes,
            ),
            *_render_batched_pdfs_for_document(
                document,
                batch_index=int(batch["batch_index"]),
                start_page=midpoint + 1,
                end_page=end_page,
                max_bytes=max_bytes,
            ),
        ]


class ExtractionChain:
    """OpenAI-backed concept extraction and relation verification (async)."""

    def __init__(self, client: StructuredExtractionClient | None = None):
        self.client = client or OpenAIResponsesClient()
        self.settings = get_settings()
        self.last_batches: list[dict[str, Any]] = []
        self.last_failed_batches: list[dict[str, Any]] = []
        self.last_usage: list[dict[str, int]] = []

    @atimeit
    async def extract_from_document(
        self,
        file_path: str,
        subject_id: str,
        page_batch_size: int | None = None,
        *,
        max_previous_concepts: int = 20,
        job_id: str | None = None,
    ) -> list[ConceptExtraction]:
        batch_size = page_batch_size or self.settings.content_pipeline_pdf_page_batch_size
        max_batch_bytes = self.settings.content_pipeline_pdf_batch_max_bytes
        self.last_batches = []
        self.last_failed_batches = []
        self.last_usage = []
        extraction_started_at = time.perf_counter()

        results: list[ConceptExtraction] = []
        previous_concepts: list[tuple[str, str]] = []
        page_count = await run_process_stage(
            _READ_PAGE_COUNT_PROCESS_TARGET,
            file_path,
            stage_name="extract_page_count",
        )
        logger.info(
            "extract start job_id={} file={} subject={} pages={} batch_size={} max_bytes={}",
            job_id or "-",
            Path(file_path).name,
            subject_id,
            page_count,
            batch_size,
            max_batch_bytes,
        )
        pending_batches: list[dict[str, Any]] = []
        page_ranges = build_page_batches(page_count, batch_size)
        for base_batch_index, (start_page, end_page) in enumerate(page_ranges):
            pending_batches.extend(
                await run_process_stage(
                    _RENDER_BATCHES_PROCESS_TARGET,
                    file_path,
                    batch_index=base_batch_index,
                    start_page=start_page,
                    end_page=end_page,
                    max_bytes=max_batch_bytes,
                    stage_name="extract_render_batch",
                )
            )

        total_planned_batches = len(pending_batches)
        completed_batches = 0
        processed_concepts = 0
        logger.info(
            "extract queue ready job_id={} total={} batch_size={} file={}",
            job_id or "-",
            total_planned_batches,
            batch_size,
            Path(file_path).name,
        )
        while pending_batches:
            rendered_batch = pending_batches.pop(0)
            try:
                extraction = await self._extract_single_batch(
                    job_id=job_id,
                    subject_id=subject_id,
                    batch=rendered_batch,
                    previous_concepts=previous_concepts[-max_previous_concepts:],
                    source_name=Path(file_path).name,
                )
            except ProviderUploadTooLargeError as exc:
                split_batches = await run_process_stage(
                    _SPLIT_BATCH_PROCESS_TARGET,
                    file_path,
                    batch=rendered_batch,
                    max_bytes=max_batch_bytes,
                    stage_name="extract_split_batch",
                )
                if split_batches:
                    total_planned_batches += len(split_batches) - 1
                    logger.warning(
                        "extract batch split retry job_id={} done={}/{} pages={} size_bytes={} reason={}",
                        job_id or "-",
                        completed_batches,
                        total_planned_batches,
                        _format_pages(
                            int(rendered_batch["page_start"]),
                            int(rendered_batch["page_end"]),
                        ),
                        rendered_batch["size_bytes"],
                        str(exc),
                    )
                    pending_batches = [*split_batches, *pending_batches]
                    continue
                extraction = ConceptExtraction(
                    concepts=[],
                    subject_id=subject_id,
                    notes=f"Error: {str(exc)[:200]}",
                )
            self.last_batches.append(rendered_batch)
            results.append(extraction)

            batch_concepts = len(getattr(extraction, "concepts", []) or [])
            processed_concepts += batch_concepts
            completed_batches += 1

            if extraction.notes and str(extraction.notes).startswith("Error:"):
                self.last_failed_batches.append(
                    {
                        "batch_index": len(results) - 1,
                        "page_start": rendered_batch["page_start"],
                        "page_end": rendered_batch["page_end"],
                        "reason": extraction.notes,
                    }
                )
            else:
                for concept in extraction.concepts:
                    concept_entry = (concept.concept_id, concept.name)
                    if concept_entry not in previous_concepts:
                        previous_concepts.append(concept_entry)

            logger.info(
                "extract progress job_id={} done={}/{} pages={} +concepts={} total_concepts={} failed={} remaining={}",
                job_id or "-",
                completed_batches,
                total_planned_batches,
                _format_pages(
                    int(rendered_batch["page_start"]),
                    int(rendered_batch["page_end"]),
                ),
                batch_concepts,
                processed_concepts,
                len(self.last_failed_batches),
                len(pending_batches),
            )
        total_concepts = sum(len(extraction.concepts) for extraction in results)
        total_usage = {
            "input_tokens": sum(item.get("input_tokens", 0) for item in self.last_usage),
            "output_tokens": sum(item.get("output_tokens", 0) for item in self.last_usage),
            "total_tokens": sum(item.get("total_tokens", 0) for item in self.last_usage),
        }
        logger.info(
            "extract done job_id={} done={}/{} concepts={} failed={} rendered_batches={} usage={} duration_ms={}",
            job_id or "-",
            completed_batches,
            total_planned_batches,
            total_concepts,
            len(self.last_failed_batches),
            len(self.last_batches),
            _format_usage(total_usage),
            int((time.perf_counter() - extraction_started_at) * 1000),
        )
        return results

    @atimeit
    async def verify_relations_batch(
        self,
        concept_pairs: list[tuple[str, str]],
        max_workers: int | None = None,
    ) -> list[EvidenceVerification]:
        worker_count = max(1, max_workers or self.settings.llm_max_workers)
        semaphore = asyncio.Semaphore(worker_count)

        async def _verify_one(pair_index: int, concept_a: str, concept_b: str) -> EvidenceVerification:
            async with semaphore:
                try:
                    return await self._verify_single_relation(concept_a, concept_b, pair_index)
                except Exception as exc:
                    logger.error("Error verifying pair {}: {}", pair_index, exc)
                    return self._verification_error(
                        f"Error during verification: {str(exc)[:100]}"
                    )

        tasks = [
            _verify_one(i, concept_a, concept_b)
            for i, (concept_a, concept_b) in enumerate(concept_pairs)
        ]
        results = await asyncio.gather(*tasks)
        return [
            result
            if result is not None
            else self._verification_error("Relation verification did not produce a result.")
            for result in results
        ]

    def _render_batched_pdfs(
        self,
        *,
        batch_index: int,
        start_page: int,
        end_page: int,
        max_bytes: int,
        file_path: str | None = None,
        document: fitz.Document | None = None,
    ) -> list[dict[str, Any]]:
        if document is None:
            if file_path is None:
                raise ValueError("Either file_path or document must be provided.")
            with fitz.open(file_path) as opened_document:
                return self._render_batched_pdfs(
                    file_path=file_path,
                    document=opened_document,
                    batch_index=batch_index,
                    start_page=start_page,
                    end_page=end_page,
                    max_bytes=max_bytes,
                )

        pdf_bytes = self._extract_pdf_bytes(document, start_page, end_page)
        rendered_batch = self._build_rendered_batch(
            batch_index=batch_index,
            start_page=start_page,
            end_page=end_page,
            pdf_bytes=pdf_bytes,
        )

        if rendered_batch["size_bytes"] > max_bytes:
            compressed_batch = self._compress_rendered_batch(
                document=document,
                batch_index=batch_index,
                start_page=start_page,
                end_page=end_page,
                original_size_bytes=rendered_batch["size_bytes"],
                max_bytes=max_bytes,
            )
            if compressed_batch is not None:
                rendered_batch = compressed_batch

        if rendered_batch["size_bytes"] <= max_bytes or start_page == end_page:
            logger.debug(
                "prepared provider batch batch={} pages={} size_bytes={} compressed={} profile={}",
                batch_index,
                _format_pages(start_page, end_page),
                rendered_batch["size_bytes"],
                rendered_batch.get("compression_applied", False),
                rendered_batch.get("compression_profile") or "-",
            )
            return [{**rendered_batch}]

        midpoint = (start_page + end_page) // 2
        logger.warning(
            "provider batch over byte limit batch={} pages={} size_bytes={} max_bytes={} split_at={} compressed={}",
            batch_index,
            _format_pages(start_page, end_page),
            rendered_batch["size_bytes"],
            max_bytes,
            midpoint,
            rendered_batch.get("compression_applied", False),
        )
        return [
            *self._render_batched_pdfs(
                document=document,
                batch_index=batch_index,
                start_page=start_page,
                end_page=midpoint,
                max_bytes=max_bytes,
            ),
            *self._render_batched_pdfs(
                document=document,
                batch_index=batch_index,
                start_page=midpoint + 1,
                end_page=end_page,
                max_bytes=max_bytes,
            ),
        ]

    @staticmethod
    def _build_rendered_batch(
        *,
        batch_index: int,
        start_page: int,
        end_page: int,
        pdf_bytes: bytes,
        compression_applied: bool = False,
        compression_profile: str | None = None,
    ) -> dict[str, Any]:
        return {
            "batch_index": batch_index,
            "page_start": start_page,
            "page_end": end_page,
            "pdf_bytes": pdf_bytes,
            "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "size_bytes": len(pdf_bytes),
            "compression_applied": compression_applied,
            "compression_profile": compression_profile,
        }

    def _compress_rendered_batch(
        self,
        *,
        document: fitz.Document,
        batch_index: int,
        start_page: int,
        end_page: int,
        original_size_bytes: int,
        max_bytes: int,
    ) -> dict[str, Any] | None:
        best_batch: dict[str, Any] | None = None
        best_size = original_size_bytes

        for dpi, jpg_quality in _COMPRESSION_PROFILES:
            compressed_bytes = self._extract_compressed_pdf_bytes(
                document,
                start_page,
                end_page,
                dpi=dpi,
                jpg_quality=jpg_quality,
            )
            compressed_batch = self._build_rendered_batch(
                batch_index=batch_index,
                start_page=start_page,
                end_page=end_page,
                pdf_bytes=compressed_bytes,
                compression_applied=True,
                compression_profile=f"{dpi}dpi-q{jpg_quality}",
            )
            if compressed_batch["size_bytes"] >= best_size:
                continue

            best_batch = compressed_batch
            best_size = compressed_batch["size_bytes"]
            logger.debug(
                "compressed provider batch candidate batch={} pages={} profile={} original_size_bytes={} compressed_size_bytes={} meets_target={}",
                batch_index,
                _format_pages(start_page, end_page),
                f"{dpi}dpi-q{jpg_quality}",
                original_size_bytes,
                best_size,
                best_size <= max_bytes,
            )
            if best_size <= max_bytes:
                break

        return best_batch

    def _split_rendered_batch(
        self,
        *,
        batch: dict[str, Any],
        max_bytes: int,
        file_path: str | None = None,
        document: fitz.Document | None = None,
    ) -> list[dict[str, Any]]:
        if document is None:
            if file_path is None:
                raise ValueError("Either file_path or document must be provided.")
            with fitz.open(file_path) as opened_document:
                return self._split_rendered_batch(
                    file_path=file_path,
                    document=opened_document,
                    batch=batch,
                    max_bytes=max_bytes,
                )

        start_page = int(batch["page_start"])
        end_page = int(batch["page_end"])
        if start_page >= end_page:
            return []
        midpoint = (start_page + end_page) // 2
        logger.warning(
            "provider batch split after upstream rejection batch={} pages={} size_bytes={} split_at={}",
            batch["batch_index"],
            _format_pages(start_page, end_page),
            batch["size_bytes"],
            midpoint,
        )
        return [
            *self._render_batched_pdfs(
                file_path=file_path,
                document=document,
                batch_index=int(batch["batch_index"]),
                start_page=start_page,
                end_page=midpoint,
                max_bytes=max_bytes,
            ),
            *self._render_batched_pdfs(
                file_path=file_path,
                document=document,
                batch_index=int(batch["batch_index"]),
                start_page=midpoint + 1,
                end_page=end_page,
                max_bytes=max_bytes,
            ),
        ]

    @staticmethod
    def _read_page_count(file_path: str) -> int:
        with fitz.open(file_path) as document:
            return int(document.page_count)

    async def _extract_single_batch(
        self,
        *,
        job_id: str | None,
        subject_id: str,
        batch: dict[str, Any],
        previous_concepts: list[tuple[str, str]],
        source_name: str,
    ) -> ConceptExtraction:
        batch_label = f"{source_name}:pages-{batch['page_start']}-{batch['page_end']}.pdf"
        pages = _format_pages(int(batch["page_start"]), int(batch["page_end"]))
        batch_started_at = time.perf_counter()
        logger.info(
            "extract batch send job_id={} batch={} pages={} size_bytes={} previous_concepts={}",
            job_id or "-",
            batch["batch_index"],
            pages,
            batch["size_bytes"],
            len(previous_concepts),
        )
        try:
            upload_started_at = time.perf_counter()
            uploaded_file = await self.client.upload_pdf_bytes(
                filename=batch_label,
                pdf_bytes=batch["pdf_bytes"],
                sha256=str(batch["sha256"]),
                now_ts=time.time(),
                job_id=job_id,
            )
            upload_duration_ms = int((time.perf_counter() - upload_started_at) * 1000)
            batch["file_id"] = uploaded_file.file_id
            batch["cache_hit"] = uploaded_file.cache_hit
            batch["purpose"] = uploaded_file.purpose
            logger.info(
                "extract batch ready job_id={} batch={} pages={} file_id={} source={} purpose={} upload_ms={}",
                job_id or "-",
                batch["batch_index"],
                pages,
                uploaded_file.file_id,
                "cache" if uploaded_file.cache_hit else "upload",
                uploaded_file.purpose,
                upload_duration_ms,
            )
            try:
                llm_started_at = time.perf_counter()
                payload, usage = await self._invoke_extraction_response_with_retries(
                    job_id=job_id,
                    subject_id=subject_id,
                    file_id=uploaded_file.file_id,
                    previous_concepts=previous_concepts,
                )
                llm_duration_ms = int((time.perf_counter() - llm_started_at) * 1000)
            except FileReferenceError:
                logger.warning(
                    "extract batch cache miss retry job_id={} batch={} pages={} file_id={} sha256={}",
                    job_id or "-",
                    batch["batch_index"],
                    pages,
                    batch.get("file_id"),
                    str(batch["sha256"])[:12],
                )
                await self.client.invalidate_cached_file(sha256=str(batch["sha256"]))
                upload_started_at = time.perf_counter()
                uploaded_file = await self.client.upload_pdf_bytes(
                    filename=batch_label,
                    pdf_bytes=batch["pdf_bytes"],
                    sha256=str(batch["sha256"]),
                    now_ts=time.time(),
                    job_id=job_id,
                )
                upload_duration_ms = int((time.perf_counter() - upload_started_at) * 1000)
                batch["file_id"] = uploaded_file.file_id
                batch["cache_hit"] = False
                batch["purpose"] = uploaded_file.purpose
                logger.info(
                    "extract batch ready job_id={} batch={} pages={} file_id={} source={} purpose={} upload_ms={}",
                    job_id or "-",
                    batch["batch_index"],
                    pages,
                    uploaded_file.file_id,
                    "reupload",
                    uploaded_file.purpose,
                    upload_duration_ms,
                )
                llm_started_at = time.perf_counter()
                payload, usage = await self._invoke_extraction_response_with_retries(
                    job_id=job_id,
                    subject_id=subject_id,
                    file_id=uploaded_file.file_id,
                    previous_concepts=previous_concepts,
                )
                llm_duration_ms = int((time.perf_counter() - llm_started_at) * 1000)
        except PayloadTooLargeError as exc:
            logger.warning(
                "extract batch upload too large job_id={} batch={} pages={} size_bytes={} reason={}",
                job_id or "-",
                batch["batch_index"],
                pages,
                batch["size_bytes"],
                str(exc),
            )
            raise ProviderUploadTooLargeError(str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "extract batch failed before structured output job_id={} batch={} pages={}",
                job_id or "-",
                batch["batch_index"],
                pages,
            )
            return ConceptExtraction(
                concepts=[],
                subject_id=subject_id,
                notes=f"Error: {str(exc)[:200]}",
            )

        try:
            self.last_usage.append(usage)
            materialized = materialize_concept_extraction(payload)
            logger.info(
                "extract batch recv job_id={} batch={} pages={} file_id={} concepts={} usage={} llm_ms={} total_ms={}",
                job_id or "-",
                batch["batch_index"],
                pages,
                batch.get("file_id"),
                len(materialized.concepts),
                _format_usage(usage),
                llm_duration_ms,
                int((time.perf_counter() - batch_started_at) * 1000),
            )
        except Exception as exc:
            logger.exception(
                "extract batch parse failed job_id={} batch={} pages={} file_id={}",
                job_id or "-",
                batch["batch_index"],
                pages,
                batch.get("file_id"),
            )
            return ConceptExtraction(
                concepts=[],
                subject_id=subject_id,
                notes=f"Error: invalid structured output ({str(exc)[:180]})",
            )
        else:
            return materialized

    async def _invoke_extraction_response(
        self,
        *,
        job_id: str | None,
        subject_id: str,
        file_id: str,
        previous_concepts: list[tuple[str, str]],
    ) -> tuple[ConceptExtractionPayload, dict[str, int]]:
        previous_section = ""
        if previous_concepts:
            previous_items = "\n".join(
                f"- `{concept_id}` : {concept_name}"
                for concept_id, concept_name in previous_concepts
            )
            previous_section = (
                "## CÁC KHÁI NIỆM ĐÃ TRÍCH XUẤT\n\n"
                "Danh sách dưới đây là các khái niệm đã có từ batch trước. "
                "Không trích xuất lại nếu cùng nghĩa. Nếu tạo relation tới chúng, "
                "phải dùng đúng `concept_id` đã cho.\n\n"
                f"{previous_items}\n\n"
            )

        user_message = (
            "## THÔNG TIN TÀI LIỆU\n"
            f"- subject_id: {subject_id}\n\n"
            f"{previous_section}"
            "Hãy đọc file PDF đính kèm và trả về đúng dữ liệu theo structured schema đã chỉ định. "
            "Không thêm văn bản ngoài schema."
        )
        logger.info(
            "extract batch llm request job_id={} file_id={} subject={} previous_concepts={}",
            job_id or "-",
            file_id,
            subject_id,
            len(previous_concepts),
        )
        response = await self.client.parse_response(
            instructions=EXTRACTION_PROMPT,
            input_blocks=[
                {"type": "input_text", "text": user_message},
                {"type": "input_file", "file_id": file_id},
            ],
            text_format=ConceptExtractionPayload,
            job_id=job_id,
        )
        return require_parsed_output(response, ConceptExtractionPayload), response_usage_summary(response)

    async def _invoke_extraction_response_with_retries(
        self,
        *,
        job_id: str | None,
        subject_id: str,
        file_id: str,
        previous_concepts: list[tuple[str, str]],
        max_retries: int | None = None,
    ) -> tuple[ConceptExtractionPayload, dict[str, int]]:
        retry_count = max(1, int(max_retries or self.settings.content_pipeline_llm_retry_attempts))
        retry_backoff_sec = max(0.0, float(self.settings.content_pipeline_llm_retry_backoff_sec))
        last_error: BaseException | None = None
        for attempt in range(retry_count):
            attempt_started_at = time.perf_counter()
            try:
                payload, usage = await self._invoke_extraction_response(
                    job_id=job_id,
                    subject_id=subject_id,
                    file_id=file_id,
                    previous_concepts=previous_concepts,
                )
                logger.info(
                    "extract batch llm success job_id={} file_id={} attempt={}/{} duration_ms={}",
                    job_id or "-",
                    file_id,
                    attempt + 1,
                    retry_count,
                    int((time.perf_counter() - attempt_started_at) * 1000),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= retry_count - 1:
                    raise
                logger.warning(
                    "extract batch retry job_id={} file_id={} attempt={}/{} reason={} duration_ms={}",
                    job_id or "-",
                    file_id,
                    attempt + 1,
                    retry_count,
                    str(exc)[:200],
                    int((time.perf_counter() - attempt_started_at) * 1000),
                )
                await asyncio.sleep(retry_backoff_sec * (attempt + 1))
            else:
                return payload, usage
        raise RuntimeError(str(last_error or "Extraction response failed."))

    async def _verify_single_relation(
        self,
        concept_a: str,
        concept_b: str,
        pair_idx: int,
        max_retries: int = 3,
    ) -> EvidenceVerification:
        user_message = (
            "## CONCEPTS TO ANALYZE\n\n"
            f"- Concept A: {concept_a}\n"
            f"- Concept B: {concept_b}\n\n"
            "Trả về đúng dữ liệu theo structured schema đã chỉ định. Không thêm văn bản ngoài schema."
        )
        last_error: BaseException | None = None
        for attempt in range(max_retries):
            try:
                response_payload = await self.client.parse_response(
                    instructions=EVIDENCE_VERIFICATION_PROMPT,
                    input_blocks=[{"type": "input_text", "text": user_message}],
                    text_format=EvidenceVerification,
                )
                return require_parsed_output(response_payload, EvidenceVerification)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Verification attempt {}/{} failed for pair {}: {}",
                    attempt + 1,
                    max_retries,
                    pair_idx,
                    exc,
                )
                await asyncio.sleep(0.5 * (attempt + 1))
        return self._verification_error(
            f"Failed to generate structured output after {max_retries} attempts. Last error: {str(last_error)[:100]}"
        )

    @staticmethod
    def _extract_pdf_bytes(
        document: fitz.Document,
        start_page: int,
        end_page: int,
    ) -> bytes:
        with fitz.open() as sub_document:
            sub_document.insert_pdf(document, from_page=start_page - 1, to_page=end_page - 1)
            return sub_document.tobytes(garbage=4, deflate=True)  # type: ignore[no-any-return]

    @staticmethod
    def _extract_compressed_pdf_bytes(
        document: fitz.Document,
        start_page: int,
        end_page: int,
        *,
        dpi: int,
        jpg_quality: int,
    ) -> bytes:
        with fitz.open() as compressed_document:
            for page_number in range(start_page - 1, end_page):
                source_page = document[page_number]
                pixmap = source_page.get_pixmap(dpi=dpi, alpha=False)
                image_bytes = pixmap.tobytes("jpeg", jpg_quality=jpg_quality)
                target_page = compressed_document.new_page(
                    width=source_page.rect.width,
                    height=source_page.rect.height,
                )
                target_page.insert_image(target_page.rect, stream=image_bytes)
            return compressed_document.tobytes(garbage=4, deflate=True)  # type: ignore[no-any-return]

    @staticmethod
    def _verification_error(reasoning: str) -> EvidenceVerification:
        return EvidenceVerification(
            has_relation=False,
            relation_type=None,
            direction=None,
            confidence=0.0,
            evidences=[],
            reasoning=reasoning,
        )
