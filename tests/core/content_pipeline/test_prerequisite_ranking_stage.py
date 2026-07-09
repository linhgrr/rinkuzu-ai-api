import asyncio
from types import SimpleNamespace

from api.domains.content_pipeline.application.stages.prerequisite_ranking import (
    rank_candidate_prerequisites,
)
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.domains.content_pipeline.domain.relations import RelationCandidate


def test_rank_candidate_prerequisites_updates_progress_and_returns_candidates():
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
        return [RelationCandidate(source_id="c1", target_id="c2", sources=frozenset({"mlp"}))]

    candidates = asyncio.run(
        rank_candidate_prerequisites(
            job,
            concepts=concepts,
            prs_threshold=0.75,
            rank_prerequisites=rank_prerequisites,
            persist_job_state=persist_job_state,
        )
    )

    assert candidates == [
        RelationCandidate(source_id="c1", target_id="c2", sources=frozenset({"mlp"}))
    ]
    assert calls == [
        (PipelineStatus.RANKING, "Ranking prerequisites...", 0.60),
        (PipelineStatus.RANKING, "Ranking prerequisites...", 0.65),
    ]
