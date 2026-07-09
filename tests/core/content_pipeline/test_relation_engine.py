import asyncio
from types import SimpleNamespace

from api.domains.content_pipeline.application.relation_engine import DefaultRelationEngine
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.domains.content_pipeline.domain.relations import RelationCandidate


def _verification(*, confidence=0.9, direction="A_to_B"):
    return SimpleNamespace(
        has_relation=True,
        relation_type="PREREQUISITE",
        confidence=confidence,
        direction=direction,
        evidences=[],
        reasoning=None,
    )


def test_default_relation_engine_merges_ranking_and_extraction_then_verifies():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha", definition="A", relations=[]),
        SimpleNamespace(
            concept_id="c2",
            name="Beta",
            definition="B",
            relations=[
                SimpleNamespace(
                    type="PREREQUISITE",
                    target_id="c1",
                    confidence=0.7,
                    evidence="Alpha is used first.",
                )
            ],
        ),
    ]
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def rank_prerequisites(items, threshold):
        assert items == concepts
        assert threshold is None
        return [
            RelationCandidate(
                source_id="c1",
                target_id="c2",
                sources=frozenset({"mlp"}),
                ranker_score=0.91,
            )
        ]

    async def verify_relations_batch(pairs):
        assert pairs == [("Alpha: A", "Beta: B\nExtracted evidence: Alpha is used first.")]
        return [_verification(confidence=0.9, direction="A_to_B")]

    engine = DefaultRelationEngine(
        rank_prerequisites=rank_prerequisites,
        verify_relations_batch=verify_relations_batch,
    )

    result = asyncio.run(
        engine.discover_relations(
            job=job,
            concepts=concepts,
            prs_threshold=None,
            min_confidence=0.6,
            persist_job_state=persist_job_state,
        )
    )

    assert [(candidate.source_id, candidate.target_id) for candidate in result.candidates] == [
        ("c1", "c2")
    ]
    assert result.candidates[0].sources == frozenset({"mlp", "extraction"})
    assert len(result.verified_relations) == 1
    assert result.verified_relations[0].source_id == "c1"
    assert result.verified_relations[0].target_id == "c2"
    assert job.relations_verified == 1
    assert calls == [
        (PipelineStatus.RANKING, "Ranking prerequisites...", 0.60),
        (PipelineStatus.RANKING, "Ranking prerequisites...", 0.65),
        (PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.70),
        (PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.80),
    ]
