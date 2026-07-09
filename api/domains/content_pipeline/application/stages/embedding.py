"""Embedding stage — DEPRECATED.

Originally produced ``Concept.name_embedding`` and ``Concept.definition_embedding``
via vietnamese-sbert for the legacy PRS ranker. After the swap to
``MLPPrerequisiteRanker`` (which carries its own BGE-M3 encoder), nothing
else in the pipeline reads those embeddings. The stage is now a no-op kept
only so existing callers in ``pipeline_runner`` and the dependency-injection
graph don't break.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def resolve_embedding_settings() -> tuple[str, int]:
    """Kept for backward-compat with ``pipeline_runner``. Returns dummy values."""
    return ("", 0)


async def compute_concept_embeddings(
    job: PipelineJob,
    *,
    concepts: list[Any],
    persist_job_state: PersistJobStateFn,
    model_name: str,
    batch_size: int,
) -> None:
    """No-op. Concept embeddings are computed inside ``MLPPrerequisiteRanker``."""
    _ = (concepts, model_name, batch_size)
    await persist_job_state(
        job,
        PipelineStatus.EMBEDDING,
        "Skipping legacy embeddings...",
        PipelineProgress.EMBEDDING_DONE,
    )
