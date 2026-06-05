"""Structured text extraction chain for content-pipeline concept work."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, cast

from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.infrastructure.prompts import (
    EVIDENCE_VERIFICATION_PROMPT,
    EXTRACTION_PROMPT,
)
from api.core.content_pipeline.infrastructure.utils.timeit import atimeit
from api.core.shared.document_text import (
    DocumentTextExtractor,
    ExtractedDocumentText,
    build_document_text_extractor,
    build_text_batches,
)
from api.core.shared.llm import make_async_llm_retry

from .schemas import (
    ConceptExtraction,
    ConceptExtractionPayload,
    EvidenceVerification,
    materialize_concept_extraction,
)
from .structured_generation import (
    LiteLLMStructuredGenerationClient,
    StructuredGenerationClient,
)


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


class ExtractionChain:
    """LiteLLM-backed concept extraction and relation verification."""

    def __init__(
        self,
        client: StructuredGenerationClient | None = None,
        document_extractor: DocumentTextExtractor | None = None,
    ) -> None:
        self.client = client or LiteLLMStructuredGenerationClient()
        self.document_extractor = document_extractor or build_document_text_extractor()
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
        document_text: ExtractedDocumentText | None = None,
        max_previous_concepts: int = 20,
        job_id: str | None = None,
    ) -> list[ConceptExtraction]:
        batch_size = page_batch_size or self.settings.content_pipeline_pdf_page_batch_size
        self.last_batches = []
        self.last_failed_batches = []
        self.last_usage = []
        extraction_started_at = time.perf_counter()

        if document_text is None:
            document_text = self.document_extractor.extract_file(file_path)
            source = "ocr"
        else:
            source = "pipeline_cache"
        pending_batches = build_text_batches(document_text.pages, batch_size=batch_size)

        page_count = int(document_text.metadata.get("page_count") or len(document_text.pages))
        results: list[ConceptExtraction] = []
        previous_concepts: list[tuple[str, str]] = []
        total_planned_batches = len(pending_batches)
        completed_batches = 0
        processed_concepts = 0

        logger.info(
            "extract start job_id={} file={} subject={} pages={} batch_size={} source={}",
            job_id or "-",
            Path(file_path).name,
            subject_id,
            page_count,
            batch_size,
            source,
        )

        while pending_batches:
            batch = pending_batches.pop(0)
            extraction = await self._extract_single_batch(
                job_id=job_id,
                subject_id=subject_id,
                batch=batch,
                previous_concepts=previous_concepts[-max_previous_concepts:],
                source_name=Path(file_path).name,
            )
            self.last_batches.append(batch)
            self.last_usage.append({})
            results.append(extraction)

            batch_concepts = len(getattr(extraction, "concepts", []) or [])
            processed_concepts += batch_concepts
            completed_batches += 1

            if extraction.notes and str(extraction.notes).startswith("Error:"):
                self.last_failed_batches.append(
                    {
                        "batch_index": len(results) - 1,
                        "page_start": batch["page_start"],
                        "page_end": batch["page_end"],
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
                _format_pages(int(batch["page_start"]), int(batch["page_end"])),
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
            "extract done job_id={} done={}/{} concepts={} failed={} batches={} usage={} duration_ms={}",
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

        async def _verify_one(
            pair_index: int,
            concept_a: str,
            concept_b: str,
        ) -> EvidenceVerification:
            async with semaphore:
                try:
                    return await self._verify_single_relation(concept_a, concept_b, pair_index)
                except Exception as exc:
                    logger.error("Error verifying pair {}: {}", pair_index, exc)
                    return self._verification_error(f"Error during verification: {str(exc)[:100]}")

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

    async def _extract_single_batch(
        self,
        *,
        job_id: str | None,
        subject_id: str,
        batch: dict[str, Any],
        previous_concepts: list[tuple[str, str]],
        source_name: str,
    ) -> ConceptExtraction:
        pages = _format_pages(int(batch["page_start"]), int(batch["page_end"]))
        batch_started_at = time.perf_counter()
        logger.info(
            "extract batch send job_id={} batch={} pages={} chars={} previous_concepts={}",
            job_id or "-",
            batch["batch_index"],
            pages,
            batch["char_count"],
            len(previous_concepts),
        )
        try:
            payload = await self._invoke_extraction_response_with_retries(
                job_id=job_id,
                subject_id=subject_id,
                document_text=str(batch["text"]),
                previous_concepts=previous_concepts,
            )
            materialized = materialize_concept_extraction(payload)
        except Exception as exc:
            logger.exception(
                "extract batch failed job_id={} batch={} pages={} source={}",
                job_id or "-",
                batch["batch_index"],
                pages,
                source_name,
            )
            return ConceptExtraction(
                concepts=[],
                subject_id=subject_id,
                notes=f"Error: {str(exc)[:200]}",
            )

        logger.info(
            "extract batch recv job_id={} batch={} pages={} concepts={} total_ms={}",
            job_id or "-",
            batch["batch_index"],
            pages,
            len(materialized.concepts),
            int((time.perf_counter() - batch_started_at) * 1000),
        )
        return materialized

    @make_async_llm_retry(label="extract batch")
    async def _invoke_extraction_response(
        self,
        *,
        job_id: str | None,
        subject_id: str,
        document_text: str,
        previous_concepts: list[tuple[str, str]],
    ) -> ConceptExtractionPayload:
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
            "<document_text>\n"
            f"{document_text}\n"
            "</document_text>\n\n"
            "Hãy đọc phần văn bản tài liệu ở trên và trả về đúng dữ liệu theo JSON/schema đã chỉ định. "
            "Không thêm văn bản ngoài schema."
        )
        logger.info(
            "extract batch llm request job_id={} subject={} previous_concepts={}",
            job_id or "-",
            subject_id,
            len(previous_concepts),
        )
        return await self.client.parse_response(
            instructions=EXTRACTION_PROMPT,
            user_text=user_message,
            text_format=ConceptExtractionPayload,
            job_id=job_id,
        )

    async def _invoke_extraction_response_with_retries(
        self,
        *,
        job_id: str | None,
        subject_id: str,
        document_text: str,
        previous_concepts: list[tuple[str, str]],
        max_retries: int | None = None,
    ) -> ConceptExtractionPayload:
        del max_retries
        return cast(
            "ConceptExtractionPayload",
            await self._invoke_extraction_response(
                job_id=job_id,
                subject_id=subject_id,
                document_text=document_text,
                previous_concepts=previous_concepts,
            ),
        )

    @make_async_llm_retry(label="relation verification")
    async def _parse_verification_response(
        self,
        user_message: str,
    ) -> EvidenceVerification:
        return cast(
            "EvidenceVerification",
            await self.client.parse_response(
                instructions=EVIDENCE_VERIFICATION_PROMPT,
                user_text=user_message,
                text_format=EvidenceVerification,
            ),
        )

    async def _verify_single_relation(
        self,
        concept_a: str,
        concept_b: str,
        pair_idx: int,
        max_retries: int = 3,  # noqa: ARG002 — kept for API compatibility; effective attempts from resolve_retry_policy()
    ) -> EvidenceVerification:
        user_message = (
            "## CONCEPTS TO ANALYZE\n\n"
            f"- Concept A: {concept_a}\n"
            f"- Concept B: {concept_b}\n\n"
            "Trả về đúng dữ liệu theo schema đã chỉ định. Không thêm văn bản ngoài schema."
        )
        try:
            return cast(
                "EvidenceVerification", await self._parse_verification_response(user_message)
            )
        except Exception as exc:
            logger.warning(
                "Verification failed for pair {} after all retries: {}",
                pair_idx,
                exc,
            )
            return self._verification_error(
                f"Failed to generate structured output after all retries. Last error: {str(exc)[:100]}"
            )

    @staticmethod
    def _verification_error(message: str) -> EvidenceVerification:
        return EvidenceVerification(
            has_relation=False,
            relation_type=None,
            direction=None,
            confidence=0.0,
            evidences=[],
            reasoning=message,
        )
