from api.core.content_pipeline import PipelineJob, PipelineStatus


def test_pipeline_status_terminal_states():
    assert PipelineStatus.COMPLETED.is_terminal is True
    assert PipelineStatus.FAILED.is_terminal is True
    assert PipelineStatus.CANCELLED.is_terminal is True
    assert PipelineStatus.EXTRACTING.is_terminal is False


def test_pipeline_job_mark_completed_sets_terminal_fields():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="math")

    job.mark_completed()

    assert job.status == PipelineStatus.COMPLETED
    assert job.progress == 1.0
    assert job.completed_at is not None


def test_pipeline_job_mark_failed_sets_status_and_message():
    job = PipelineJob(job_id="job-2", filename="lesson.pdf", subject_id="math")

    job.mark_failed("boom")

    assert job.status == PipelineStatus.FAILED
    assert job.error_message == "boom"
