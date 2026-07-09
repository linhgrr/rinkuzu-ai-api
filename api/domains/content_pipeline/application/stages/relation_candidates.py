"""Relation candidate extraction and merge helpers."""

from __future__ import annotations

from typing import Any

from api.domains.content_pipeline.domain.relations import RelationCandidate

MLP_SOURCE = "mlp"
EXTRACTION_SOURCE = "extraction"


def _clean_evidence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    seen: set[str] = set()
    evidences: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            evidences.append(text)
    return tuple(evidences)


def extract_relation_candidates(concepts: list[Any]) -> tuple[list[RelationCandidate], int]:
    """Convert extracted concept relations into verifier candidates.

    Extracted PREREQUISITE relations are attached to the dependent concept and
    point at its prerequisite. Candidate edges use graph direction:
    prerequisite source -> dependent target.
    """
    concept_ids = {
        str(getattr(concept, "concept_id", "")).strip()
        for concept in concepts
        if getattr(concept, "concept_id", None)
    }
    candidates: list[RelationCandidate] = []
    dropped = 0

    for concept in concepts:
        dependent_id = str(getattr(concept, "concept_id", "")).strip()
        if not dependent_id:
            continue
        for relation in getattr(concept, "relations", []) or []:
            relation_type = str(getattr(relation, "type", "")).strip().upper()
            prerequisite_id = str(getattr(relation, "target_id", "")).strip()
            if (
                relation_type != "PREREQUISITE"
                or not prerequisite_id
                or prerequisite_id == dependent_id
                or prerequisite_id not in concept_ids
            ):
                dropped += 1
                continue
            confidence = getattr(relation, "confidence", None)
            candidates.append(
                RelationCandidate(
                    source_id=prerequisite_id,
                    target_id=dependent_id,
                    sources=frozenset({EXTRACTION_SOURCE}),
                    extraction_confidence=float(confidence) if confidence is not None else None,
                    extracted_evidences=_clean_evidence(getattr(relation, "evidence", None)),
                )
            )

    return merge_relation_candidates(candidates), dropped


def merge_relation_candidates(candidates: list[RelationCandidate]) -> list[RelationCandidate]:
    """Dedupe candidates by directed edge while preserving source metadata."""
    merged: dict[tuple[str, str], RelationCandidate] = {}

    for candidate in candidates:
        key = (candidate.source_id, candidate.target_id)
        current = merged.get(key)
        if current is None:
            merged[key] = candidate
            continue

        evidences = tuple(
            dict.fromkeys(current.extracted_evidences + candidate.extracted_evidences)
        )
        ranker_score = (
            max(current.ranker_score, candidate.ranker_score)
            if current.ranker_score is not None and candidate.ranker_score is not None
            else current.ranker_score
            if current.ranker_score is not None
            else candidate.ranker_score
        )
        extraction_confidence = (
            max(current.extraction_confidence, candidate.extraction_confidence)
            if current.extraction_confidence is not None
            and candidate.extraction_confidence is not None
            else current.extraction_confidence
            if current.extraction_confidence is not None
            else candidate.extraction_confidence
        )
        merged[key] = RelationCandidate(
            source_id=current.source_id,
            target_id=current.target_id,
            sources=current.sources | candidate.sources,
            ranker_score=ranker_score,
            extraction_confidence=extraction_confidence,
            extracted_evidences=evidences,
        )

    return list(merged.values())
