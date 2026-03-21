import asyncio
from types import SimpleNamespace

from api.core.content_pipeline.application.stages.concept_merge import merge_duplicate_concepts
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_merge_duplicate_concepts_updates_job_metrics_and_partial_graph():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def merge_by_name(items):
        assert items == concepts
        return [items[1]]

    merged = asyncio.run(
        merge_duplicate_concepts(
            job,
            concepts=concepts,
            merge_by_name=merge_by_name,
            persist_job_state=persist_job_state,
        )
    )

    assert merged == [concepts[1]]
    assert job.concepts_after_merge == 1
    assert job.partial_graph == {
        "nodes": [{"id": "c2", "name": "Beta"}],
        "edges": [],
    }
    assert calls == [
        (PipelineStatus.MERGING, "Merging duplicate concepts...", 0.50),
        (PipelineStatus.MERGING, "Merging duplicate concepts...", 0.55),
    ]
