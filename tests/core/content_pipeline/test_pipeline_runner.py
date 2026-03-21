from api.core.content_pipeline.application.pipeline_runner import (
    PipelineRunner,
    populate_job_metrics_from_result,
)
from api.core.content_pipeline.domain.jobs import PipelineJob


def test_populate_job_metrics_from_result_derives_summary_fields():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    job.result = {
        "concept_map": {"c1": 0, "c2": 1},
        "prereq_edges": [{"source": "c1", "target": "c2"}],
        "stats": {"num_nodes": 4, "num_edges": 3, "relations_verified": 7, "is_dag": False},
    }

    populate_job_metrics_from_result(job)

    assert job.concepts_extracted == 2
    assert job.concepts_after_merge == 4
    assert job.relations_verified == 7
    assert job.graph_stats == {
        "num_nodes": 4,
        "num_edges": 3,
        "relations_verified": 7,
        "is_dag": False,
    }


def test_pipeline_runner_keeps_constructor_dependencies():
    async def load_job(job_id: str):
        return None

    async def save_job(job):
        return True

    async def persist_job_state(job, status, step, progress):
        return None

    runner = PipelineRunner(
        load_job=load_job,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )

    assert runner._load_job is load_job
    assert runner._save_job is save_job
    assert runner._persist_job_state is persist_job_state
