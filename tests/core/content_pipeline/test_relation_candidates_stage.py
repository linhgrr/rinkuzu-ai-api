from types import SimpleNamespace

from api.domains.content_pipeline.application.stages.relation_candidates import (
    extract_relation_candidates,
    merge_relation_candidates,
)
from api.domains.content_pipeline.domain.relations import RelationCandidate


def test_extract_relation_candidates_converts_extracted_relation_direction():
    concepts = [
        SimpleNamespace(
            concept_id="dependent",
            relations=[
                SimpleNamespace(
                    type="PREREQUISITE",
                    target_id="prereq",
                    confidence=0.8,
                    evidence="Needs prereq first.",
                ),
                SimpleNamespace(type="PREREQUISITE", target_id="missing", confidence=0.9),
            ],
        ),
        SimpleNamespace(concept_id="prereq", relations=[]),
    ]

    candidates, dropped = extract_relation_candidates(concepts)

    assert dropped == 1
    assert candidates == [
        RelationCandidate(
            source_id="prereq",
            target_id="dependent",
            sources=frozenset({"extraction"}),
            extraction_confidence=0.8,
            extracted_evidences=("Needs prereq first.",),
        )
    ]


def test_merge_relation_candidates_preserves_sources_and_scores():
    merged = merge_relation_candidates(
        [
            RelationCandidate(
                source_id="c1",
                target_id="c2",
                sources=frozenset({"mlp"}),
                ranker_score=0.7,
            ),
            RelationCandidate(
                source_id="c1",
                target_id="c2",
                sources=frozenset({"extraction"}),
                extraction_confidence=0.9,
                extracted_evidences=("e1",),
            ),
        ]
    )

    assert merged == [
        RelationCandidate(
            source_id="c1",
            target_id="c2",
            sources=frozenset({"mlp", "extraction"}),
            ranker_score=0.7,
            extraction_confidence=0.9,
            extracted_evidences=("e1",),
        )
    ]
