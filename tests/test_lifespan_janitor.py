"""Tests for PipelineJanitor wiring in the FastAPI app lifespan."""

from __future__ import annotations

import types
from typing import Any

import pytest

from api import main as main_module
from api.config import get_settings


class _RecordingJanitor:
    """Test double standing in for PipelineJanitor.

    Records the construction kwargs and whether start/stop were awaited.
    """

    def __init__(
        self,
        *,
        recover: Any,
        reap: Any,
        reaper_interval_sec: int,
    ) -> None:
        self.recover = recover
        self.reap = reap
        self.reaper_interval_sec = reaper_interval_sec
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _FakeService:
    """Records calls to the recover/reap service methods."""

    def __init__(self) -> None:
        self.recover_kwargs: dict[str, Any] | None = None
        self.reap_kwargs: dict[str, Any] | None = None

    async def recover_interrupted_jobs(self, **kwargs: Any) -> None:
        self.recover_kwargs = kwargs

    async def reap_stalled_jobs(self, **kwargs: Any) -> int:
        self.reap_kwargs = kwargs
        return 0


def test_build_pipeline_janitor_wires_settings_and_closures() -> None:
    """_build_pipeline_janitor builds a PipelineJanitor with the right interval
    and closures that invoke the service with the configured settings."""
    settings = get_settings()
    service = _FakeService()

    janitor = main_module._build_pipeline_janitor(service, settings)

    from api.core.content_pipeline.application.recovery import PipelineJanitor

    assert isinstance(janitor, PipelineJanitor)
    # The janitor stores the interval (clamped to a minimum of 10s internally).
    assert janitor._interval == max(10, settings.content_pipeline_reaper_interval_sec)


@pytest.mark.asyncio
async def test_build_pipeline_janitor_closures_call_service() -> None:
    """The recover/reap closures forward the right injected deps + settings."""
    settings = get_settings()
    service = _FakeService()

    janitor = main_module._build_pipeline_janitor(service, settings)

    await janitor._recover()
    await janitor._reap()

    assert service.recover_kwargs is not None
    assert service.recover_kwargs["list_active"] is main_module.list_active_pipeline_jobs
    assert service.recover_kwargs["download_source"] is main_module.download_source_to_dir
    assert (
        service.recover_kwargs["recovery_max_age_sec"]
        == settings.content_pipeline_recovery_max_age_sec
    )

    assert service.reap_kwargs is not None
    assert service.reap_kwargs["list_active"] is main_module.list_active_pipeline_jobs
    assert (
        service.reap_kwargs["stalled_after_sec"] == settings.content_pipeline_job_stalled_after_sec
    )


def test_init_pipeline_sets_janitor_on_app_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """_init_pipeline constructs the janitor (the double) and stores it on app.state
    with the configured reaper interval."""
    monkeypatch.setattr(main_module, "PipelineJanitor", _RecordingJanitor)

    settings = get_settings()
    app = types.SimpleNamespace(state=types.SimpleNamespace(chunk_chroma_store=object()))

    main_module._init_pipeline(app)  # type: ignore[arg-type]

    janitor = app.state.pipeline_janitor
    assert isinstance(janitor, _RecordingJanitor)
    assert janitor.reaper_interval_sec == settings.content_pipeline_reaper_interval_sec
    # Closures are wired and awaitable.
    assert callable(janitor.recover)
    assert callable(janitor.reap)


@pytest.mark.asyncio
async def test_janitor_double_start_stop_are_awaitable() -> None:
    """The janitor double's start/stop flip flags when awaited (mirrors lifespan use)."""
    janitor = _RecordingJanitor(recover=lambda: None, reap=lambda: None, reaper_interval_sec=60)
    assert not janitor.started
    assert not janitor.stopped
    await janitor.start()
    await janitor.stop()
    assert janitor.started
    assert janitor.stopped
