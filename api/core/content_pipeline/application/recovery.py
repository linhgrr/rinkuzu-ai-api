"""Background janitor: startup recovery + periodic stalled-job reaper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class PipelineJanitor:
    def __init__(
        self,
        *,
        recover: Callable[[], Awaitable[None]],
        reap: Callable[[], Awaitable[None]],
        reaper_interval_sec: int,
    ) -> None:
        self._recover = recover
        self._reap = reap
        self._interval = max(10, reaper_interval_sec)
        self._scheduler = AsyncIOScheduler()

    async def _run_reaper(self) -> None:
        try:
            await self._reap()
        except Exception:
            logger.exception("[PipelineJanitor] reaper run failed")

    async def start(self) -> None:
        try:
            await self._recover()
        except Exception:
            logger.exception("[PipelineJanitor] startup recovery failed")
        self._scheduler.add_job(
            self._run_reaper,
            "interval",
            seconds=self._interval,
            id="pipeline_reaper",
            replace_existing=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info("[PipelineJanitor] started (reaper every {}s)", self._interval)

    async def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
