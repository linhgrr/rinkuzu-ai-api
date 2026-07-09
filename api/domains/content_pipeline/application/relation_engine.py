"""Relation engine abstractions for the content pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ports import PersistJobStateFn, RelationDiscoveryResult
from .stages.prerequisite_ranking import rank_candidate_prerequisites
from .stages.relation_candidates import extract_relation_candidates, merge_relation_candidates
from .stages.relation_verification import RelationVerification, verify_candidate_relations

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from api.domains.content_pipeline.domain.jobs import PipelineJob
    from api.domains.content_pipeline.domain.relations import RelationCandidate


class DefaultRelationEngine:
    """Default relation engine that preserves the current ranking + verification flow."""

    def __init__(
        self,
        *,
        rank_prerequisites: Callable[[list[Any], float | None], list[RelationCandidate]],
        verify_relations_batch: Callable[
            [list[tuple[str, str]]],
            Awaitable[list[RelationVerification]],
        ],
    ) -> None:
        self._rank_prerequisites = rank_prerequisites
        self._verify_relations_batch = verify_relations_batch

    async def discover_relations(
        self,
        *,
        job: PipelineJob,
        concepts: list[Any],
        prs_threshold: float | None,
        min_confidence: float,
        persist_job_state: PersistJobStateFn,
    ) -> RelationDiscoveryResult:
        ranked_candidates = await rank_candidate_prerequisites(
            job,
            concepts=concepts,
            prs_threshold=prs_threshold,
            rank_prerequisites=self._rank_prerequisites,
            persist_job_state=persist_job_state,
        )
        extracted_candidates, dropped_extracted = extract_relation_candidates(concepts)
        candidates = merge_relation_candidates(ranked_candidates + extracted_candidates)
        if dropped_extracted:
            job.graph_stats["relations_extraction_candidates_dropped"] = dropped_extracted

        verified_relations = await verify_candidate_relations(
            job,
            concepts=concepts,
            candidates=candidates,
            min_confidence=min_confidence,
            verify_relations_batch=self._verify_relations_batch,
            persist_job_state=persist_job_state,
        )
        return RelationDiscoveryResult(
            candidates=candidates,
            verified_relations=verified_relations,
        )
