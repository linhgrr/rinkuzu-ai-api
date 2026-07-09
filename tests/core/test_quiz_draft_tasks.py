import asyncio

import pytest

from api.domains.quiz import draft_tasks as draft_tasks_module
from api.domains.quiz.draft_tasks import QuizDraftTaskManager


class FakeService:
    def __init__(self, calls: list[tuple[str, str]], release: asyncio.Event) -> None:
        self._calls = calls
        self._release = release

    async def process_draft(self, draft_id: str, user_id: str) -> None:
        self._calls.append((draft_id, user_id))
        await self._release.wait()


@pytest.mark.asyncio
async def test_task_manager_deduplicates_and_shuts_down_cleanly():
    calls: list[tuple[str, str]] = []
    release = asyncio.Event()
    manager = QuizDraftTaskManager(lambda: FakeService(calls, release))

    assert manager.schedule("draft-1", "user-1") is True
    assert manager.schedule("draft-1", "user-1") is False
    await asyncio.sleep(0)

    assert calls == [("draft-1", "user-1")]

    release.set()
    await asyncio.sleep(0)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_task_manager_recovers_queued_and_processing_drafts(monkeypatch):
    calls: list[tuple[str, str]] = []
    release = asyncio.Event()
    manager = QuizDraftTaskManager(lambda: FakeService(calls, release))

    async def fake_list_recoverable_quiz_drafts():
        return [
            {"draft_id": "draft-1", "user_id": "user-1"},
            {"draft_id": "draft-2", "user_id": "user-2"},
        ]

    monkeypatch.setattr(
        draft_tasks_module,
        "list_recoverable_quiz_drafts",
        fake_list_recoverable_quiz_drafts,
    )

    assert await manager.recover() == 2
    await asyncio.sleep(0)
    assert calls == [("draft-1", "user-1"), ("draft-2", "user-2")]

    release.set()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_task_manager_limits_concurrency_and_cancels_draft():
    calls: list[tuple[str, str]] = []
    release = asyncio.Event()
    manager = QuizDraftTaskManager(
        lambda: FakeService(calls, release),
        max_concurrent_jobs=1,
    )

    manager.schedule("draft-1", "user-1")
    manager.schedule("draft-2", "user-2")
    await asyncio.sleep(0)

    assert calls == [("draft-1", "user-1")]

    await manager.cancel("draft-1")
    await asyncio.sleep(0)
    assert calls == [("draft-1", "user-1"), ("draft-2", "user-2")]

    release.set()
    await manager.shutdown()
