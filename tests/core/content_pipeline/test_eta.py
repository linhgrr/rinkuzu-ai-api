from api.core.content_pipeline.application.eta import estimate_eta_seconds
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus


def _job(progress: float, pages: int = 10) -> PipelineJob:
    j = PipelineJob(job_id="j", filename="a.pdf", subject_id="a")
    j.total_pages = pages
    j.progress = progress
    return j


def test_eta_zero_when_complete():
    j = _job(PipelineProgress.COMPLETE)
    j.status = PipelineStatus.COMPLETED
    assert estimate_eta_seconds(j, secs_per_page=20.0) == 0.0


def test_eta_scales_with_remaining_progress():
    early = estimate_eta_seconds(_job(0.15), secs_per_page=20.0)
    late = estimate_eta_seconds(_job(0.85), secs_per_page=20.0)
    assert early is not None
    assert late is not None
    assert early > late > 0.0


def test_eta_none_when_no_pages_yet():
    j = _job(0.0, pages=0)
    j.status = PipelineStatus.QUEUED
    assert estimate_eta_seconds(j, secs_per_page=20.0) is None
