"""Ports used by the unified content pipeline application layer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

PersistJobStateFn = Callable[
    [PipelineJob, PipelineStatus, str, float],
    Awaitable[None],
]


@dataclass(frozen=True)
class RelationDiscoveryResult:
    """Stable output contract for relation discovery implementations."""

    candidate_pairs: list[tuple[str, str]]
    verified_relations: list[tuple[str, str, Any]]


class RelationEngine(Protocol):
    """Application-facing contract for relation discovery algorithms."""

    async def discover_relations(
        self,
        *,
        job: PipelineJob,
        concepts: list[Any],
        prs_threshold: float,
        min_confidence: float,
        persist_job_state: PersistJobStateFn,
    ) -> RelationDiscoveryResult: ...
