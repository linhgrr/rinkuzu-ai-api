import asyncio
from types import SimpleNamespace

from api.domains.content_pipeline.application.stages.relation_verification import (
    build_pairs_to_verify,
    filter_verified_relations,
    verify_candidate_relations,
)
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.domains.content_pipeline.domain.relations import RelationCandidate


def _verification(*, has_relation=True, confidence=0.9, direction="A_to_B"):
    return SimpleNamespace(
        has_relation=has_relation,
        relation_type="PREREQUISITE" if has_relation else None,
        confidence=confidence,
        direction=direction,
        evidences=[],
        reasoning=None,
    )


def test_build_pairs_to_verify_maps_concept_ids_to_names():
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha", definition="Definition A"),
        SimpleNamespace(concept_id="c2", name="Beta", definition=""),
    ]
    candidates = [
        RelationCandidate(source_id="c1", target_id="c2", sources=frozenset({"mlp"})),
        RelationCandidate(
            source_id="c2",
            target_id="missing",
            sources=frozenset({"extraction"}),
            extracted_evidences=("e1",),
        ),
    ]

    pairs = build_pairs_to_verify(concepts, candidates)

    assert pairs == [("Alpha: Definition A", "Beta"), ("Beta", "missing\nExtracted evidence: e1")]


def test_filter_verified_relations_applies_relation_and_confidence_checks():
    candidates = [
        RelationCandidate(source_id="c1", target_id="c2", sources=frozenset({"mlp"})),
        RelationCandidate(source_id="c2", target_id="c3", sources=frozenset({"mlp"})),
        RelationCandidate(source_id="c3", target_id="c4", sources=frozenset({"mlp"})),
    ]
    verifications = [
        _verification(confidence=0.9, direction="A_to_B"),
        _verification(has_relation=False, confidence=0.99, direction="A_to_B"),
        _verification(confidence=0.4, direction="B_to_A"),
    ]

    verified = filter_verified_relations(
        candidates,
        verifications,
        min_confidence=0.6,
    )

    assert len(verified) == 1
    assert verified[0].source_id == "c1"
    assert verified[0].target_id == "c2"
    assert verified[0].confidence == 0.9


def test_filter_verified_relations_applies_reverse_direction():
    candidate = RelationCandidate(source_id="c1", target_id="c2", sources=frozenset({"mlp"}))
    verified = filter_verified_relations(
        [candidate],
        [_verification(confidence=0.8, direction="B_to_A")],
        min_confidence=0.6,
    )

    assert len(verified) == 1
    assert verified[0].source_id == "c2"
    assert verified[0].target_id == "c1"


def test_verify_candidate_relations_updates_progress_and_job_metrics():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha", definition=""),
        SimpleNamespace(concept_id="c2", name="Beta", definition=""),
    ]
    calls: list[tuple[PipelineStatus, str, float]] = []
    verifier_calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    async def verify_relations_batch(pairs):
        verifier_calls.append(pairs)
        return [_verification(confidence=0.8, direction="A_to_B")]

    verified = asyncio.run(
        verify_candidate_relations(
            job,
            concepts=concepts,
            candidates=[
                RelationCandidate(source_id="c1", target_id="c2", sources=frozenset({"mlp"}))
            ],
            min_confidence=0.6,
            verify_relations_batch=verify_relations_batch,
            persist_job_state=persist_job_state,
        )
    )

    assert verifier_calls == [[("Alpha", "Beta")]]
    assert len(verified) == 1
    assert job.relations_verified == 1
    assert calls == [
        (PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.70),
        (PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.80),
    ]
