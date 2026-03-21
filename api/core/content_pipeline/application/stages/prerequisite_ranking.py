"""Prerequisite ranking stage for the content pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ...domain.jobs import PipelineJob, PipelineStatus
from .execution import run_blocking_stage


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def rank_candidate_prerequisites(
    job: PipelineJob,
    *,
    concepts: list[Any],
    prs_threshold: float,
    rank_prerequisites: Callable[[list[Any], float], list[tuple[str, str]]],
    persist_job_state: PersistJobStateFn,
) -> list[tuple[str, str]]:
    """Rank candidate prerequisite pairs and persist stage progress."""
    await persist_job_state(job, PipelineStatus.RANKING, "Ranking prerequisites...", 0.60)

    candidate_pairs = await run_blocking_stage(
        rank_prerequisites,
        concepts,
        prs_threshold,
        stage_name="prerequisite_ranking",
    )

    await persist_job_state(job, PipelineStatus.RANKING, "Ranking prerequisites...", 0.65)
    return candidate_pairs
