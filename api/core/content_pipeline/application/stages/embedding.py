"""Embedding stage for the content pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ...domain.jobs import PipelineJob, PipelineStatus


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def resolve_embedding_settings() -> tuple[str, int]:
    """Preserve current embedding config fallback while isolating it from orchestration."""
    try:
        from config import settings as cp_settings

        return cp_settings.embedding_model, cp_settings.embedding_batch_size
    except ImportError:
        return "keepitreal/vietnamese-sbert", 32


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
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        compute_embedding_for_concepts,
        concepts,
        embed_client,
    )

    await persist_job_state(job, PipelineStatus.EMBEDDING, "Computing embeddings...", 0.45)
