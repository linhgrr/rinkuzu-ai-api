"""Concept merge stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from .concept_extraction import build_partial_concept_graph
from .execution import run_blocking_stage

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def merge_duplicate_concepts(
    job: PipelineJob,
    *,
    concepts: list[Any],
    merge_by_name: Callable[[list[Any]], list[Any]],
    persist_job_state: PersistJobStateFn,
) -> list[Any]:
    """Merge duplicate concepts and persist stage progress."""
    await persist_job_state(job, PipelineStatus.MERGING, "Merging duplicate concepts...", 0.50)

    merged_concepts = await run_blocking_stage(
        merge_by_name,
        concepts,
        stage_name="concept_merge",
    )
    job.concepts_after_merge = len(merged_concepts)
    job.partial_graph = build_partial_concept_graph(merged_concepts)

    await persist_job_state(job, PipelineStatus.MERGING, "Merging duplicate concepts...", 0.55)
    return merged_concepts
