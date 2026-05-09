from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.core.content_pipeline.infrastructure.serializers import pipeline_job_to_document


def test_pipeline_job_to_document_matches_repository_shape():
    job = PipelineJob(
        job_id="job-1",
        filename="lesson.pdf",
        subject_id="math",
        user_id="user-1",
        status=PipelineStatus.EXTRACTING,
        current_step="Extracting concepts",
        progress=0.25,
        total_chunks=4,
        page_batch_size=10,
        batch_count=3,
        failed_batch_count=1,
        partial_success=True,
        concepts_extracted=10,
        concepts_after_merge=8,
        relations_verified=2,
        graph_stats={"num_nodes": 8},
        error_message=None,
        error_code=None,
        user_message=None,
        retryable=False,
        result={"concept_map": {}},
        partial_graph={"nodes": [], "edges": []},
        created_at=123.0,
        completed_at=None,
    )

    doc = pipeline_job_to_document(job)

    assert doc["job_id"] == "job-1"
    assert doc["status"] == "extracting"
    assert doc["current_step"] == "Extracting concepts"
    assert doc["progress"] == 0.25
    assert doc["page_batch_size"] == 10
    assert doc["batch_count"] == 3
    assert doc["failed_batch_count"] == 1
    assert doc["partial_success"] is True
    assert doc["created_at"] == 123.0
    assert doc["completed_at"] is None
