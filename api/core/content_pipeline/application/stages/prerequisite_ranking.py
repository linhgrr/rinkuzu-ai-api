"""Prerequisite ranking stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from .execution import run_blocking_stage

if TYPE_CHECKING:
    from api.core.content_pipeline.domain.relations import RelationCandidate

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


async def rank_candidate_prerequisites(
    job: PipelineJob,
    *,
    concepts: list[Any],
    prs_threshold: float | None,
    rank_prerequisites: Callable[[list[Any], float | None], list[RelationCandidate]],
    persist_job_state: PersistJobStateFn,
) -> list[RelationCandidate]:
    """Rank candidate prerequisite pairs and persist stage progress."""
    await persist_job_state(
        job, PipelineStatus.RANKING, "Ranking prerequisites...", PipelineProgress.RANKING_START
    )

    candidates: list[RelationCandidate] = await run_blocking_stage(
        rank_prerequisites,
        concepts,
        prs_threshold,
        stage_name="prerequisite_ranking",
    )

    await persist_job_state(
        job, PipelineStatus.RANKING, "Ranking prerequisites...", PipelineProgress.RANKING_DONE
    )
    return candidates
