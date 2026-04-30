"""Embedding stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from api.config import get_settings
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from .execution import run_blocking_stage

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def resolve_embedding_settings() -> tuple[str, int]:
    """Read embedding settings from the unified backend config."""
    settings = get_settings()
    return settings.embedding_model, settings.embedding_batch_size


async def compute_concept_embeddings(
    job: PipelineJob,
    *,
    concepts: list[Any],
    embedding_client_factory: Callable[[str, int], Any],
    compute_embedding_for_concepts: Callable[[list[Any], Any], Any],
    persist_job_state: PersistJobStateFn,
    model_name: str,
    batch_size: int,
) -> None:
    """Compute embeddings for extracted concepts and persist stage progress."""
    await persist_job_state(job, PipelineStatus.EMBEDDING, "Computing embeddings...", 0.35)

    embed_client = embedding_client_factory(model_name, batch_size)
    await run_blocking_stage(
        compute_embedding_for_concepts,
        concepts,
        embed_client,
        stage_name="embedding",
    )

    await persist_job_state(job, PipelineStatus.EMBEDDING, "Computing embeddings...", 0.45)
