"""Relation verification stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]
VerifiedRelation = tuple[str, str, Any]


def build_pairs_to_verify(
    concepts: list[Any],
    candidate_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Map candidate concept ids into the human-readable names sent to the verifier."""
    concept_name_map = {
        str(getattr(concept, "concept_id", "")): str(getattr(concept, "name", ""))
        for concept in concepts
    }
    return [
        (concept_name_map.get(source_id, source_id), concept_name_map.get(target_id, target_id))
        for source_id, target_id in candidate_pairs
    ]


def filter_verified_relations(
    candidate_pairs: list[tuple[str, str]],
    verifications: list[Any],
    *,
    min_confidence: float,
) -> list[VerifiedRelation]:
    """Keep only relations that passed verifier checks and confidence threshold."""
    verified: list[VerifiedRelation] = []
    for (source_id, target_id), evaluation in zip(candidate_pairs, verifications, strict=False):
        if evaluation and evaluation.has_relation and evaluation.confidence >= min_confidence:
            verified.append((source_id, target_id, evaluation))
    return verified


async def verify_candidate_relations(
    job: PipelineJob,
    *,
    concepts: list[Any],
    candidate_pairs: list[tuple[str, str]],
    min_confidence: float,
    verify_relations_batch: Callable[[list[tuple[str, str]]], Awaitable[list[Any]]],
    persist_job_state: PersistJobStateFn,
) -> list[VerifiedRelation]:
    """Verify prerequisite candidates with the LLM verifier and persist stage progress."""
    await persist_job_state(
        job,
        PipelineStatus.VERIFYING,
        "Verifying relations with LLM...",
        PipelineProgress.RELATION_VERIFICATION_START,
    )

    pairs_to_verify = build_pairs_to_verify(concepts, candidate_pairs)
    verified: list[VerifiedRelation] = []
    if pairs_to_verify:
        verifications: list[Any] = await verify_relations_batch(pairs_to_verify)
        verified = filter_verified_relations(
            candidate_pairs,
            verifications,
            min_confidence=min_confidence,
        )

    job.relations_verified = len(verified)
    await persist_job_state(
        job,
        PipelineStatus.VERIFYING,
        "Verifying relations with LLM...",
        PipelineProgress.RELATION_VERIFICATION_DONE,
    )
    return verified
