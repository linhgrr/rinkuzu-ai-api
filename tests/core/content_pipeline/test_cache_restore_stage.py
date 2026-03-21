import asyncio

from api.core.content_pipeline.application.stages.cache_restore import (
    try_restore_completed_job_from_mongo,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


async def _load_completed_job(job_id: str):
    return {
        "job_id": job_id,
        "filename": "lesson.pdf",
        "subject_id": "algebra",
        "status": PipelineStatus.COMPLETED.value,
        "result": {
            "concept_map": {"c1": 0},
            "prereq_edges": [{"source": "c1", "target": "c1"}],
            "stats": {"relations_verified": 1},
        },
        "total_chunks": 4,
        "concepts_extracted": 2,
        "concepts_after_merge": 1,
        "relations_verified": 1,
        "graph_stats": {"num_nodes": 1},
        "completed_at": 123.0,
    }


def test_try_restore_completed_job_from_mongo_populates_job_state():
    job = PipelineJob(job_id="job-1", filename="upload.pdf", subject_id="math")

    restored = asyncio.run(
        try_restore_completed_job_from_mongo(
            job,
            load_job=_load_completed_job,
            populate_metrics=lambda restored_job: None,
        )
    )

    assert restored is True
    assert job.status == PipelineStatus.COMPLETED
    assert job.filename == "lesson.pdf"
    assert job.subject_id == "algebra"
    assert job.total_chunks == 4
    assert job.progress == 1.0
    assert job.current_step == "Loaded from MongoDB"


async def _load_missing_job(_: str):
    return None


def test_try_restore_completed_job_from_mongo_returns_false_for_miss():
    job = PipelineJob(job_id="job-2", filename="upload.pdf", subject_id="math")

    restored = asyncio.run(
        try_restore_completed_job_from_mongo(
            job,
            load_job=_load_missing_job,
            populate_metrics=lambda restored_job: None,
        )
    )

    assert restored is False
    assert job.status == PipelineStatus.PENDING
