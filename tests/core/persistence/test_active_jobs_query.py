import inspect

from api.core.shared.persistence import pipeline_jobs


def test_active_query_helpers_exist_with_expected_signatures():
    assert hasattr(pipeline_jobs, "list_active_pipeline_jobs")
    assert hasattr(pipeline_jobs, "list_recent_pipeline_jobs_all_status")
    sig = inspect.signature(pipeline_jobs.list_active_pipeline_jobs)
    assert sig.parameters["user_id"].default is None


def test_non_terminal_statuses_excludes_terminal():
    from api.core.content_pipeline.domain.jobs import PipelineStatus

    statuses = set(pipeline_jobs._NON_TERMINAL_STATUSES)
    assert PipelineStatus.COMPLETED.value not in statuses
    assert PipelineStatus.FAILED.value not in statuses
    assert PipelineStatus.CANCELLED.value not in statuses
    assert PipelineStatus.EXTRACTING.value in statuses
