import asyncio
from types import SimpleNamespace

from api.core.content_pipeline.application.stages.relation_verification import (
    build_pairs_to_verify,
    filter_verified_relations,
    verify_candidate_relations,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_build_pairs_to_verify_maps_concept_ids_to_names():
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]

    pairs = build_pairs_to_verify(concepts, [("c1", "c2"), ("c2", "missing")])

    assert pairs == [("Alpha", "Beta"), ("Beta", "missing")]


def test_filter_verified_relations_applies_relation_and_confidence_checks():
    candidate_pairs = [("c1", "c2"), ("c2", "c3"), ("c3", "c4")]
    verifications = [
        SimpleNamespace(has_relation=True, confidence=0.9, direction="A_to_B"),
        SimpleNamespace(has_relation=False, confidence=0.99, direction="A_to_B"),
        SimpleNamespace(has_relation=True, confidence=0.4, direction="B_to_A"),
    ]

    verified = filter_verified_relations(
        candidate_pairs,
        verifications,
        min_confidence=0.6,
    )

    assert verified == [("c1", "c2", verifications[0])]


def test_verify_candidate_relations_updates_progress_and_job_metrics():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    calls: list[tuple[PipelineStatus, str, float]] = []
    verifier_calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    async def verify_relations_batch(pairs):
        verifier_calls.append(pairs)
        return [SimpleNamespace(has_relation=True, confidence=0.8, direction="A_to_B")]

    verified = asyncio.run(
        verify_candidate_relations(
            job,
            concepts=concepts,
            candidate_pairs=[("c1", "c2")],
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
