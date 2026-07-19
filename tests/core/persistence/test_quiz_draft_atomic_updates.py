"""Behavior/query-shape proofs for quiz draft owner-scoped atomic updates."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from api.shared.persistence import quiz_drafts as quiz_draft_store
from api.shared.persistence.documents import QuizDraftStatus


class _EqField:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return (self.name, "==", other)

    def __hash__(self) -> int:
        return hash(self.name)


class _FakeFind:
    def __init__(self, *, update_result=None, update_side_effect=None):
        self._update_result = update_result
        self._update_side_effect = update_side_effect
        self.update_calls: list[tuple[tuple, dict]] = []

    async def update(self, *args, **kwargs):
        self.update_calls.append((args, kwargs))
        if self._update_side_effect is not None:
            raise self._update_side_effect
        return self._update_result


class _AwaitableValue:
    """Stand-in for Beanie's awaitable ``find_one`` when no chained update is used."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _resolve():
            return self._value

        return _resolve().__await__()


class _FakeQuizDraftDocument:
    draft_id = _EqField("draft_id")
    user_id = _EqField("user_id")
    last_find_args: ClassVar[tuple] = ()
    last_find_kwargs: ClassVar[dict] = {}
    finder: ClassVar[_FakeFind | None] = None
    find_one_side_effect: ClassVar[Exception | None] = None
    find_one_result: ClassVar[object | None] = None

    @classmethod
    def find_one(cls, *args, **kwargs):
        cls.last_find_args = args
        cls.last_find_kwargs = kwargs
        if cls.find_one_side_effect is not None:
            raise cls.find_one_side_effect
        if cls.finder is not None:
            return cls.finder
        return _AwaitableValue(cls.find_one_result)


def _public_doc(**overrides):
    base = SimpleNamespace(
        draft_id="draft-1",
        user_id="user-1",
        title="t",
        description="",
        category_id=None,
        prompt=None,
        pdf={},
        status=QuizDraftStatus.PROCESSING,
        progress={},
        questions=[],
        error=None,
        submitted_quiz_id=None,
        created_at=None,
        updated_at=None,
        expires_at=None,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


@pytest.fixture
def fake_doc(monkeypatch):
    monkeypatch.setattr(quiz_draft_store, "QuizDraftDocument", _FakeQuizDraftDocument)
    _FakeQuizDraftDocument.last_find_args = ()
    _FakeQuizDraftDocument.last_find_kwargs = {}
    _FakeQuizDraftDocument.finder = None
    _FakeQuizDraftDocument.find_one_side_effect = None
    _FakeQuizDraftDocument.find_one_result = None
    return _FakeQuizDraftDocument


@pytest.mark.asyncio
async def test_update_query_includes_owner_and_status_ne_cancelled(fake_doc):
    doc = _public_doc()
    finder = _FakeFind(update_result=doc)
    fake_doc.finder = finder

    result = await quiz_draft_store.update_quiz_draft_for_user(
        "draft-1",
        "user-1",
        {"status": "processing", "error": None},
    )

    assert result is not None
    assert ("draft_id", "==", "draft-1") in fake_doc.last_find_args
    assert ("user_id", "==", "user-1") in fake_doc.last_find_args
    assert {"status": {"$ne": QuizDraftStatus.CANCELLED.value}} in fake_doc.last_find_args
    assert finder.update_calls
    set_payload = finder.update_calls[0][0][0]["$set"]
    assert set_payload["status"] == QuizDraftStatus.PROCESSING.value


@pytest.mark.asyncio
async def test_request_cancel_uses_owner_scoped_set(fake_doc):
    doc = _public_doc(status=QuizDraftStatus.CANCELLED)
    finder = _FakeFind(update_result=doc)
    fake_doc.finder = finder

    result = await quiz_draft_store.request_cancel_quiz_draft_for_user("draft-1", "user-1")
    assert result is not None
    assert result["status"] == QuizDraftStatus.CANCELLED.value
    assert ("user_id", "==", "user-1") in fake_doc.last_find_args
    assert finder.update_calls[0][0][0]["$set"]["status"] == QuizDraftStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_update_db_error_propagates(fake_doc):
    fake_doc.finder = _FakeFind(update_side_effect=RuntimeError("mongo down"))
    with pytest.raises(RuntimeError, match="mongo down"):
        await quiz_draft_store.update_quiz_draft_for_user("d1", "u1", {"status": "failed"})


@pytest.mark.asyncio
async def test_load_quiz_draft_db_error_propagates(fake_doc):
    fake_doc.find_one_side_effect = RuntimeError("mongo down")
    with pytest.raises(RuntimeError, match="mongo down"):
        await quiz_draft_store.load_quiz_draft_for_user("d1", "u1")


@pytest.mark.asyncio
async def test_load_quiz_draft_returns_none_only_for_absence(fake_doc):
    fake_doc.find_one_result = None
    assert await quiz_draft_store.load_quiz_draft_for_user("d1", "u1") is None

    fake_doc.find_one_result = _public_doc()
    result = await quiz_draft_store.load_quiz_draft_for_user("draft-1", "user-1")
    assert result is not None
    assert result["draft_id"] == "draft-1"
    assert ("user_id", "==", "user-1") in fake_doc.last_find_args
