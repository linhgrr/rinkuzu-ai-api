"""Relation verification stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus
from api.core.content_pipeline.domain.relations import RelationCandidate, VerifiedRelation

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


class RelationVerification(Protocol):
    has_relation: bool
    relation_type: Literal["PREREQUISITE", "SAME_CONCEPT"] | None
    direction: Literal["A_to_B", "B_to_A", "same_concept"] | None
    confidence: float
    evidences: list[str]
    reasoning: str | None


def _concept_label(concept: Any) -> str:
    name = str(getattr(concept, "name", "") or "").strip()
    definition = str(getattr(concept, "definition", "") or "").strip()
    if name and definition:
        return f"{name}: {definition}"
    return name or definition


def build_pairs_to_verify(
    concepts: list[Any],
    candidates: list[RelationCandidate],
) -> list[tuple[str, str]]:
    """Map candidate concept ids into verifier labels with definition/evidence context."""
    concept_label_map = {
        str(getattr(concept, "concept_id", "")): _concept_label(concept) for concept in concepts
    }
    pairs: list[tuple[str, str]] = []
    for candidate in candidates:
        source_label = concept_label_map.get(candidate.source_id, candidate.source_id)
        target_label = concept_label_map.get(candidate.target_id, candidate.target_id)
        if candidate.extracted_evidences:
            evidence = " | ".join(candidate.extracted_evidences[:3])
            target_label = f"{target_label}\nExtracted evidence: {evidence}"
        pairs.append((source_label, target_label))
    return pairs


def filter_verified_relations(
    candidates: list[RelationCandidate],
    verifications: list[RelationVerification],
    *,
    min_confidence: float,
) -> list[VerifiedRelation]:
    """Keep only relations that passed verifier checks and confidence threshold."""
    verified: list[VerifiedRelation] = []
    for candidate, evaluation in zip(candidates, verifications, strict=False):
        if not evaluation:
            continue
        if not evaluation.has_relation or evaluation.confidence < min_confidence:
            continue
        if evaluation.relation_type not in (None, "PREREQUISITE"):
            continue

        source_id = candidate.source_id
        target_id = candidate.target_id
        if evaluation.direction == "B_to_A":
            source_id, target_id = target_id, source_id
        elif evaluation.direction != "A_to_B":
            continue

        evidences = tuple(
            dict.fromkeys(
                candidate.extracted_evidences
                + tuple(str(item).strip() for item in getattr(evaluation, "evidences", []) if item)
            )
        )
        verified.append(
            VerifiedRelation(
                source_id=source_id,
                target_id=target_id,
                confidence=float(evaluation.confidence),
                evidences=evidences,
                reasoning=getattr(evaluation, "reasoning", None),
                sources=candidate.sources,
                ranker_score=candidate.ranker_score,
                extraction_confidence=candidate.extraction_confidence,
            )
        )
    return verified


async def verify_candidate_relations(
    job: PipelineJob,
    *,
    concepts: list[Any],
    candidates: list[RelationCandidate],
    min_confidence: float,
    verify_relations_batch: Callable[
        [list[tuple[str, str]]],
        Awaitable[list[RelationVerification]],
    ],
    persist_job_state: PersistJobStateFn,
) -> list[VerifiedRelation]:
    """Verify prerequisite candidates with the LLM verifier and persist stage progress."""
    await persist_job_state(
        job,
        PipelineStatus.VERIFYING,
        "Verifying relations with LLM...",
        PipelineProgress.RELATION_VERIFICATION_START,
    )

    pairs_to_verify = build_pairs_to_verify(concepts, candidates)
    verified: list[VerifiedRelation] = []
    if pairs_to_verify:
        verifications: list[Any] = await verify_relations_batch(pairs_to_verify)
        verified = filter_verified_relations(
            candidates,
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
