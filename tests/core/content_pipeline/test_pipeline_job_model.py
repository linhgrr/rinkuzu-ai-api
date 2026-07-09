from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def _job() -> PipelineJob:
    return PipelineJob(job_id="j1", filename="a.pdf", subject_id="a")


def test_new_fields_default():
    j = _job()
    assert j.source_s3_key is None
    assert j.retry_count == 0
    assert j.cancel_requested is False
    assert j.eta_seconds is None


def test_reset_for_retry_clears_error_and_increments_count():
    j = _job()
    j.mark_failed("boom")
    j.error_code = "pipeline_timeout"
    j.user_message = "took too long"
    j.retryable = True
    j.reset_for_retry()
    assert j.status is PipelineStatus.QUEUED
    assert j.error_message is None
    assert j.error_code is None
    assert j.user_message is None
    assert j.retry_count == 1
    assert j.cancel_requested is False
    assert j.completed_at is None


def test_request_cancel_sets_flag():
    j = _job()
    j.request_cancel()
    assert j.cancel_requested is True
