import pytest

from api.domains.content_pipeline.application.recovery import PipelineJanitor


@pytest.mark.asyncio
async def test_janitor_start_runs_recovery_once_and_schedules_reaper():
    events = {"recovery": 0, "reaper": 0}

    async def recover():
        events["recovery"] += 1

    async def reap():
        events["reaper"] += 1

    janitor = PipelineJanitor(recover=recover, reap=reap, reaper_interval_sec=60)
    await janitor.start()
    assert events["recovery"] == 1  # recovery runs immediately on start
    await janitor._run_reaper()  # trigger the scheduled job body manually
    assert events["reaper"] == 1
    await janitor.stop()


@pytest.mark.asyncio
async def test_janitor_stop_is_safe_when_not_started():
    janitor = PipelineJanitor(recover=_noop, reap=_noop, reaper_interval_sec=60)
    await janitor.stop()  # must not raise


async def _noop():
    return None


@pytest.mark.asyncio
async def test_janitor_recovery_failure_does_not_crash_start():
    async def recover():
        raise RuntimeError("boom")

    async def reap():
        return None

    janitor = PipelineJanitor(recover=recover, reap=reap, reaper_interval_sec=60)
    await janitor.start()  # should swallow the recovery error and still start
    await janitor.stop()
