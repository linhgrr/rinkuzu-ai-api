from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.shared.persistence.pipeline_jobs import pipeline_job_to_document


def test_to_document_includes_source_and_retry_fields():
    job = PipelineJob(job_id="j1", filename="a.pdf", subject_id="a")
    job.source_s3_key = "uploads/quiz_extract/u1/a.pdf"
    job.retry_count = 2
    job.cancel_requested = True
    job.eta_seconds = 42.0
    job.status = PipelineStatus.EXTRACTING
    doc = pipeline_job_to_document(job)
    assert doc["source_s3_key"] == "uploads/quiz_extract/u1/a.pdf"
    assert doc["retry_count"] == 2
    assert doc["cancel_requested"] is True
    assert doc["eta_seconds"] == 42.0
    assert doc["prs_threshold"] is None
    assert doc["min_confidence"] == 0.6
    assert doc["apply_reduction"] is True
