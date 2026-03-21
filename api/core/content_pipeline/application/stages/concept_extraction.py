"""Concept extraction stage for the content pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ...domain.jobs import PipelineJob, PipelineStatus


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


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
    chunks: list[Any],
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

    chunk_texts = [chunk.page_content for chunk in chunks]
    loop = asyncio.get_running_loop()
    extractions = await loop.run_in_executor(
        None,
        extraction_chain.extract_from_batch,
        chunk_texts,
        job.subject_id,
    )

    all_concepts: list[Any] = []
    for extraction in extractions:
        if extraction and hasattr(extraction, "concepts"):
            all_concepts.extend(extraction.concepts)

    all_concepts = postprocess_concepts(all_concepts)
    job.concepts_extracted = len(all_concepts)
    job.partial_graph = build_partial_concept_graph(all_concepts)

    await persist_job_state(
        job,
        PipelineStatus.EXTRACTING,
        "Extracting concepts with LLM...",
        0.30,
    )
    return all_concepts
