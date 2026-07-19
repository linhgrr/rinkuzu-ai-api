"""Ports used by the unified content pipeline application layer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.domains.content_pipeline.domain.transitions import CreateJobOutcome, SaveJobOutcome

if TYPE_CHECKING:
    from api.domains.content_pipeline.domain.relations import RelationCandidate, VerifiedRelation

PersistJobStateFn = Callable[
    [PipelineJob, PipelineStatus, str, float],
    Awaitable[None],
]
SaveJobFn = Callable[[PipelineJob], Awaitable[SaveJobOutcome]]
CreateJobFn = Callable[[PipelineJob], Awaitable[CreateJobOutcome]]
LoadJobFn = Callable[[str], Awaitable[dict | None]]
LoadCancelFlagFn = Callable[[str], Awaitable[bool]]


def raise_for_save_outcome(job: PipelineJob, outcome: SaveJobOutcome, *, operation: str) -> None:
    """Map non-APPLIED save outcomes to cooperative-cancel or stale-worker stops.

    APPLIED is a no-op. Infrastructure failures must be raised by the save port
    itself (never disguised as a soft outcome).
    """
    if outcome is SaveJobOutcome.APPLIED:
        return
    if outcome is SaveJobOutcome.CANCEL_REQUESTED:
        from .cancellation import JobCancelledError

        job.cancel_requested = True
        raise JobCancelledError(f"Job {job.job_id} cancel requested while {operation}")
    if outcome in (SaveJobOutcome.STALE_GENERATION, SaveJobOutcome.ALREADY_TERMINAL):
        from api.domains.content_pipeline.domain.errors import PipelineStaleWorkerError

        raise PipelineStaleWorkerError(job.job_id, outcome.value)
    raise RuntimeError(
        f"Unexpected save outcome {outcome!r} for job {job.job_id} while {operation}"
    )


__all__ = [
    "CreateJobFn",
    "CreateJobOutcome",
    "LoadCancelFlagFn",
    "LoadJobFn",
    "PersistJobStateFn",
    "RelationDiscoveryResult",
    "RelationEngine",
    "SaveJobFn",
    "SaveJobOutcome",
    "raise_for_save_outcome",
]


@dataclass(frozen=True)
class RelationDiscoveryResult:
    """Stable output contract for relation discovery implementations."""

    candidates: list[RelationCandidate]
    verified_relations: list[VerifiedRelation]


class RelationEngine(Protocol):
    """Application-facing contract for relation discovery algorithms."""

    async def discover_relations(
        self,
        *,
        job: PipelineJob,
        concepts: list[Any],
        prs_threshold: float | None,
        min_confidence: float,
        persist_job_state: PersistJobStateFn,
    ) -> RelationDiscoveryResult:
        raise NotImplementedError
