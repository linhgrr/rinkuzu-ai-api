import asyncio
from datetime import UTC, datetime

import pytest

from api.core.quiz.draft_service import (
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)
from api.core.shared import mongo_store


class _QuizDraftRepoStub:
    def __init__(self):
        self.docs = {}

    async def create(self, doc):
        self.docs[doc["draft_id"]] = doc
        return doc

    async def load_for_user(self, draft_id, user_id):
        doc = self.docs.get(draft_id)
        if doc and doc["user_id"] == user_id:
            return doc
        return None

    async def update_for_user(self, draft_id, user_id, updates):
        doc = await self.load_for_user(draft_id, user_id)
        if not doc:
            return None
        doc.update(updates)
        return doc

    async def list_recent_for_user(self, user_id, limit):
        return [doc for doc in self.docs.values() if doc["user_id"] == user_id][:limit]

    async def delete_for_user(self, draft_id, user_id):
        doc = await self.load_for_user(draft_id, user_id)
        if doc:
            self.docs.pop(draft_id)
        return doc


def test_quiz_draft_s3_key_must_belong_to_user():
    with pytest.raises(QuizDraftValidationError):
        QuizDraftService._normalize_and_validate_s3_key(
            "uploads/quiz_extract/user-2/file.pdf",
            "user-1",
        )


def test_public_draft_uses_safe_defaults():
    now = datetime.now(UTC)

    draft = public_draft(
        {
            "draft_id": "draft-1",
            "title": "Quiz",
            "status": "queued",
            "pdf": {"s3_key": "uploads/quiz_extract/user-1/file.pdf"},
            "created_at": now,
            "updated_at": now,
            "expires_at": now,
        }
    )

    assert draft["draft_id"] == "draft-1"
    assert draft["questions"] == []
    assert draft["progress"] == {"processed": 0, "total": 1, "percent": 0}


def test_mongo_store_quiz_draft_repo_helpers(monkeypatch):
    repo = _QuizDraftRepoStub()
    monkeypatch.setitem(mongo_store._state, "quiz_draft_repo", repo)
    doc = {"draft_id": "draft-1", "user_id": "user-1", "status": "queued"}

    created = asyncio.run(mongo_store.get_quiz_draft_repo().create(doc))
    loaded = asyncio.run(mongo_store.get_quiz_draft_repo().load_for_user("draft-1", "user-1"))
    updated = asyncio.run(
        mongo_store.get_quiz_draft_repo().update_for_user("draft-1", "user-1", {"status": "completed"})
    )
    listed = asyncio.run(mongo_store.get_quiz_draft_repo().list_recent_for_user("user-1", 20))
    deleted = asyncio.run(mongo_store.get_quiz_draft_repo().delete_for_user("draft-1", "user-1"))

    assert created == doc
    assert loaded == doc
    assert updated["status"] == "completed"
    assert listed == [updated]
    assert deleted == updated
