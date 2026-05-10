"""Concept extraction stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Any

import fitz
from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from ..ports import PersistJobStateFn  # noqa: TC001
from .execution import run_blocking_stage


def build_partial_concept_graph(concepts: list[Any]) -> dict[str, list[dict[str, str]]]:
    """Build the lightweight partial graph payload used during extraction."""
    return {
        "nodes": [
            {
                "id": str(getattr(concept, "concept_id", "")),
                "name": str(getattr(concept, "name", "")),
            }
            for concept in concepts
        ],
        "edges": [],
    }


async def extract_concepts_from_chunks(
    job: PipelineJob,
    *,
    file_path: str,
    extraction_chain: Any,
    postprocess_concepts: Callable[[list[Any]], list[Any]],
    persist_job_state: PersistJobStateFn,
) -> list[Any]:
    """Extract concepts from loaded document chunks and persist stage progress."""
    await persist_job_state(
        job,
        PipelineStatus.EXTRACTING,
        "Extracting concepts with LLM...",
        0.15,
    )

    settings = get_settings()
    extraction_timeout = _resolve_extraction_timeout(file_path, job, settings)

    extractions: list[Any] = await run_blocking_stage(
        extraction_chain.extract_from_document,
        file_path,
        job.subject_id,
        job.page_batch_size,
        job_id=job.job_id,
        stage_name="concept_extraction",
        timeout_sec=extraction_timeout,
    )

    all_concepts: list[Any] = []
    failed_batch_count = 0
    for extraction in extractions:
        if extraction and hasattr(extraction, "concepts"):
            all_concepts.extend(extraction.concepts)
        note = str(getattr(extraction, "notes", "") or "")
        if note.startswith("Error:"):
            failed_batch_count += 1

    all_concepts = postprocess_concepts(all_concepts)
    failed_batch_details = list(getattr(extraction_chain, "last_failed_batches", []))
    job.batch_count = len(getattr(extraction_chain, "last_batches", [])) or len(extractions)
    job.failed_batch_count = failed_batch_count or len(failed_batch_details)
    job.concepts_extracted = len(all_concepts)
    job.partial_graph = build_partial_concept_graph(all_concepts)
    if job.batch_count > 0:
        failure_ratio = job.failed_batch_count / job.batch_count
        job.partial_success = job.failed_batch_count > 0 and (
            failure_ratio <= settings.content_pipeline_batch_failure_ratio_threshold
        )

    await persist_job_state(
        job,
        PipelineStatus.EXTRACTING,
        "Extracting concepts with LLM...",
        0.30,
    )
    return all_concepts


def _resolve_extraction_timeout(file_path: str, job: PipelineJob, settings: Any) -> float:
    """Calculate extraction timeout based on PDF page count.

    Falls back to content_pipeline_stage_timeout_sec if page count cannot be read.
    Formula: max(stage_timeout, n_pages * secs_per_page)
    """
    from .execution import resolve_timeout_policy  # noqa: PLC0415

    _, default_stage_timeout = resolve_timeout_policy()
    fallback = default_stage_timeout or 300.0

    secs_per_page = float(getattr(settings, "content_pipeline_extraction_secs_per_page", 20.0))
    try:
        with fitz.open(file_path) as doc:
            n_pages = int(doc.page_count)
        job.total_pages = n_pages
    except Exception:
        logger.warning(
            "[concept_extraction] Could not open PDF to count pages, using default timeout: {}",
            fallback,
        )
        return fallback

    if n_pages <= 0:
        return fallback

    dynamic = n_pages * secs_per_page
    return max(fallback, dynamic)
