"""Ports used by the unified content pipeline application layer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

if TYPE_CHECKING:
    from api.domains.content_pipeline.domain.relations import RelationCandidate, VerifiedRelation

PersistJobStateFn = Callable[
    [PipelineJob, PipelineStatus, str, float],
    Awaitable[None],
]
SaveJobFn = Callable[[PipelineJob], Awaitable[bool]]
LoadJobFn = Callable[[str], Awaitable[dict | None]]
LoadCancelFlagFn = Callable[[str], Awaitable[bool]]


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
