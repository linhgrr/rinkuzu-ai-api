from datetime import UTC, datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_current_user
from api.domains.quiz import router as quiz_router
from api.domains.quiz.draft_service import QuizDraftConflictError, QuizDraftService
from api.domains.quiz.schemas import QuizDraftPatchRequest, QuizDraftQuestion
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.limiter = limiter
    app.include_router(quiz_router.drafts_router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(quiz_router, "is_admin_request", lambda *args, **kwargs: True)


@pytest.mark.asyncio
async def test_manual_draft_create_does_not_require_ai_dependencies(
    monkeypatch: pytest.MonkeyPatch,
):
    now = datetime.now(UTC)
    persisted: list[dict[str, object]] = []

    monkeypatch.setattr("api.domains.quiz.draft_service.mongo_store.is_available", lambda: True)

    async def fake_create(doc: dict[str, object]):
        persisted.append(doc)
        return doc

    monkeypatch.setattr("api.domains.quiz.draft_service.create_quiz_draft", fake_create)
    request = quiz_router.QuizManualDraftCreateRequest(title="  New quiz  ", is_private=True)

    result = await QuizDraftService().create_manual_draft(request, "user-1")

    assert result["status"] == "drafting"
    assert result["source_type"] == "manual"
    assert result["expires_at"] is None
    assert result["is_private"] is True
    assert result["created_at"] >= now
    assert persisted == [result]


def test_manual_draft_route_never_schedules_ai_worker(monkeypatch: pytest.MonkeyPatch):
    now = datetime.now(UTC)
    draft = {
        "draft_id": "draft-1",
        "title": "New quiz",
        "description": "",
        "category_id": None,
        "prompt": None,
        "source_type": "manual",
        "is_private": False,
        "revision": 0,
        "question_count": 0,
        "pdf": {},
        "status": "drafting",
        "progress": {"processed": 0, "total": 0, "percent": 0},
        "questions": [],
        "error": None,
        "submitted_quiz_id": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": None,
    }
    service = AsyncMock()
    service.create_manual_draft.return_value = draft
    monkeypatch.setattr(quiz_router, "QuizDraftService", lambda: service)
    schedule = AsyncMock()
    monkeypatch.setattr(quiz_router.quiz_draft_task_manager, "schedule", schedule)

    response = _client().post("/api/v1/quiz/drafts/manual", json={"title": "New quiz"})

    assert response.status_code == 200
    assert response.json()["data"]["draft"]["source_type"] == "manual"
    schedule.assert_not_called()


def test_patch_revision_conflict_maps_to_409(monkeypatch: pytest.MonkeyPatch):
    service = AsyncMock()
    service.patch_draft.side_effect = QuizDraftConflictError("Draft changed in another session.")
    monkeypatch.setattr(quiz_router, "QuizDraftService", lambda: service)

    response = _client().patch(
        "/api/v1/quiz/drafts/draft-1",
        json={"title": "Changed", "expected_revision": 1},
    )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_patch_detects_stale_revision_without_masking_missing_draft(
    monkeypatch: pytest.MonkeyPatch,
):
    service = QuizDraftService()
    monkeypatch.setattr(
        service,
        "get_draft",
        AsyncMock(return_value={"draft_id": "draft-1", "status": "drafting", "revision": 3}),
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.update_quiz_draft_for_user",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.load_quiz_draft_for_user",
        AsyncMock(return_value={"draft_id": "draft-1", "revision": 4}),
    )

    with pytest.raises(QuizDraftConflictError):
        await service.patch_draft(
            "draft-1",
            "user-1",
            QuizDraftPatchRequest(title="Changed", expected_revision=3),
        )


def test_manual_question_contract_accepts_incomplete_two_option_draft():
    question = QuizDraftQuestion.model_validate(
        {
            "question": "",
            "type": "multiple",
            "options": ["", ""],
            "correctIndexes": [],
            "optionImages": [None, None],
        }
    )

    assert question.options == ["", ""]
