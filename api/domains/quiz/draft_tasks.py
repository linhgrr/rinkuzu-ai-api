"""Application-owned task lifecycle for quiz draft extraction."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from api.core.shared.persistence.quiz_drafts import list_recoverable_quiz_drafts

from .draft_service import QuizDraftService

if TYPE_CHECKING:
    from collections.abc import Callable


class QuizDraftProcessor(Protocol):
    async def process_draft(self, draft_id: str, user_id: str) -> None: ...


class QuizDraftTaskManager:
    def __init__(
        self,
        service_factory: Callable[[], QuizDraftProcessor] = QuizDraftService,
        max_concurrent_jobs: int = 2,
    ) -> None:
        self._service_factory = service_factory
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_jobs))
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def schedule(self, draft_id: str, user_id: str) -> bool:
        existing = self._tasks.get(draft_id)
        if existing is not None and not existing.done():
            return False

        task = asyncio.create_task(
            self._run(draft_id, user_id),
            name=f"quiz-draft:{draft_id}",
        )
        self._tasks[draft_id] = task

        def on_done(completed: asyncio.Task[None]) -> None:
            self._on_done(draft_id, completed)

        task.add_done_callback(on_done)
        return True

    async def cancel(self, draft_id: str) -> None:
        task = self._tasks.get(draft_id)
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def recover(self) -> int:
        drafts = await list_recoverable_quiz_drafts()
        scheduled = 0
        for draft in drafts:
            draft_id = str(draft.get("draft_id") or "")
            user_id = str(draft.get("user_id") or "")
            if draft_id and user_id and self.schedule(draft_id, user_id):
                scheduled += 1
        if scheduled:
            logger.info("[quiz_draft] recovery_scheduled count={}", scheduled)
        return scheduled

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run(self, draft_id: str, user_id: str) -> None:
        async with self._semaphore:
            await self._service_factory().process_draft(draft_id, user_id)

    def _on_done(self, draft_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(draft_id) is task:
            self._tasks.pop(draft_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("[quiz_draft] task_crashed draft_id={} error={}", draft_id, error)


quiz_draft_task_manager = QuizDraftTaskManager()
