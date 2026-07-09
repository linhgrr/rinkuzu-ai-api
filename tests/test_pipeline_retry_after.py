from types import SimpleNamespace

from api.domains.content_pipeline import PipelineStatus
from api.domains.content_pipeline import router as pipeline


def test_pipeline_retry_after_uses_stage_specific_settings(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "get_settings",
        lambda: SimpleNamespace(
            content_pipeline_default_retry_after_sec=3,
            content_pipeline_active_retry_after_sec=5,
            content_pipeline_long_stage_retry_after_sec=10,
            content_pipeline_delayed_retry_after_sec=15,
        ),
    )

    assert (
        pipeline._resolve_pipeline_retry_after_seconds(
            status_value=PipelineStatus.QUEUED.value,
            is_terminal=False,
            is_delayed=False,
        )
        == 3
    )
    assert (
        pipeline._resolve_pipeline_retry_after_seconds(
            status_value=PipelineStatus.EMBEDDING.value,
            is_terminal=False,
            is_delayed=False,
        )
        == 5
    )
    assert (
        pipeline._resolve_pipeline_retry_after_seconds(
            status_value=PipelineStatus.EXTRACTING.value,
            is_terminal=False,
            is_delayed=False,
        )
        == 10
    )
    assert (
        pipeline._resolve_pipeline_retry_after_seconds(
            status_value=PipelineStatus.EXTRACTING.value,
            is_terminal=False,
            is_delayed=True,
        )
        == 15
    )
    assert (
        pipeline._resolve_pipeline_retry_after_seconds(
            status_value=PipelineStatus.COMPLETED.value,
            is_terminal=True,
            is_delayed=False,
        )
        == 0
    )
