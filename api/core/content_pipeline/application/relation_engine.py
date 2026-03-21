"""Relation engine abstractions for the content pipeline."""

from __future__ import annotations

from typing import Any, Callable

from .ports import PersistJobStateFn, RelationDiscoveryResult
from .stages.prerequisite_ranking import rank_candidate_prerequisites
from .stages.relation_verification import verify_candidate_relations
from ..domain.jobs import PipelineJob


class DefaultRelationEngine:
    """Default relation engine that preserves the current ranking + verification flow."""

    def __init__(
        self,
        *,
        rank_prerequisites: Callable[[list[Any], float], list[tuple[str, str]]],
        verify_relations_batch: Callable[[list[tuple[str, str]]], list[Any]],
    ) -> None:
        self._rank_prerequisites = rank_prerequisites
        self._verify_relations_batch = verify_relations_batch

    async def discover_relations(
        self,
        *,
        job: PipelineJob,
        concepts: list[Any],
        prs_threshold: float,
        min_confidence: float,
        persist_job_state: PersistJobStateFn,
    ) -> RelationDiscoveryResult:
        candidate_pairs = await rank_candidate_prerequisites(
            job,
            concepts=concepts,
            prs_threshold=prs_threshold,
            rank_prerequisites=self._rank_prerequisites,
            persist_job_state=persist_job_state,
        )
        verified_relations = await verify_candidate_relations(
            job,
            concepts=concepts,
            candidate_pairs=candidate_pairs,
            min_confidence=min_confidence,
            verify_relations_batch=self._verify_relations_batch,
            persist_job_state=persist_job_state,
        )
        return RelationDiscoveryResult(
            candidate_pairs=candidate_pairs,
            verified_relations=verified_relations,
        )
