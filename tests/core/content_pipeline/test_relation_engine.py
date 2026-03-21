import asyncio
from types import SimpleNamespace

from api.core.content_pipeline.application.relation_engine import DefaultRelationEngine
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_default_relation_engine_combines_ranking_and_verification():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def rank_prerequisites(items, threshold):
        assert items == concepts
        assert threshold == 0.75
        return [("c1", "c2")]

    def verify_relations_batch(pairs):
        assert pairs == [("Alpha", "Beta")]
        return [SimpleNamespace(has_relation=True, confidence=0.9, direction="A_to_B")]

    engine = DefaultRelationEngine(
        rank_prerequisites=rank_prerequisites,
        verify_relations_batch=verify_relations_batch,
    )

    result = asyncio.run(
        engine.discover_relations(
            job=job,
            concepts=concepts,
            prs_threshold=0.75,
            min_confidence=0.6,
            persist_job_state=persist_job_state,
        )
    )

    assert result.candidate_pairs == [("c1", "c2")]
    assert len(result.verified_relations) == 1
    assert job.relations_verified == 1
    assert calls == [
        (PipelineStatus.RANKING, "Ranking prerequisites...", 0.60),
        (PipelineStatus.RANKING, "Ranking prerequisites...", 0.65),
        (PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.70),
        (PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.80),
    ]
