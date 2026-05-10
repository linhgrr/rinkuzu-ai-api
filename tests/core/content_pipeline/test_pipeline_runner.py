import asyncio
from types import SimpleNamespace

from api.core.content_pipeline.application.pipeline_runner import (
    PipelineRunner,
    _resolve_effective_job_timeout,
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


def test_resolve_effective_job_timeout_exceeds_extraction_timeout(monkeypatch):
    job = PipelineJob(
        job_id="job-timeout",
        filename="lesson.pdf",
        subject_id="algebra",
        page_batch_size=10,
    )

    async def fake_resolve_extraction_timeout(file_path, job_arg, settings):
        del file_path, settings
        assert job_arg is job
        return 2400.0

    monkeypatch.setattr(
        "api.core.content_pipeline.application.pipeline_runner._resolve_extraction_timeout",
        fake_resolve_extraction_timeout,
    )
    monkeypatch.setattr(
        "api.core.content_pipeline.application.pipeline_runner.resolve_timeout_policy",
        lambda: (1800.0, 300.0),
    )

    timeout = asyncio.run(
        _resolve_effective_job_timeout(
            file_path="/tmp/lesson.pdf",  # noqa: S108
            job=job,
            settings=SimpleNamespace(),
        )
    )

    assert timeout == 3000.0
