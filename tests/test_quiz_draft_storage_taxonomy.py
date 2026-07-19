"""Quiz draft routes: storage infra → 503 retryable; empty 200; programmer → 500."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pymongo.errors import ServerSelectionTimeoutError
import pytest

from api.dependencies import get_current_user
from api.domains.quiz import router as quiz_router
from api.domains.quiz.draft_service import QuizDraftNotFoundError, QuizDraftValidationError
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quiz_router, "is_admin_request", lambda *_a, **_k: True)
    was = limiter.enabled
    limiter.enabled = False
    yield
    limiter.enabled = was


def _app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = limiter
    register_exception_handlers(app)
    app.include_router(quiz_router.drafts_router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return app


@pytest.mark.asyncio
async def test_list_quiz_drafts_storage_outage_503(monkeypatch: pytest.MonkeyPatch):
    service = AsyncMock()
    service.list_drafts = AsyncMock(side_effect=ServerSelectionTimeoutError("mongo down"))

    class _Svc:
        async def list_drafts(self, user_id: str, limit: int = 20):
            raise ServerSelectionTimeoutError("mongo down")

    monkeypatch.setattr(quiz_router, "QuizDraftService", _Svc)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/quiz/drafts")
    assert response.status_code == 503
    assert response.json()["error"]["meta"]["retryable"] is True


@pytest.mark.asyncio
async def test_list_quiz_drafts_empty_200(monkeypatch: pytest.MonkeyPatch):
    class _Svc:
        async def list_drafts(self, user_id: str, limit: int = 20):
            return []

    monkeypatch.setattr(quiz_router, "QuizDraftService", _Svc)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/quiz/drafts")
    assert response.status_code == 200
    assert response.json()["data"]["drafts"] == []


@pytest.mark.asyncio
async def test_get_quiz_draft_not_found_404(monkeypatch: pytest.MonkeyPatch):
    class _Svc:
        async def get_draft(self, draft_id: str, user_id: str):
            raise QuizDraftNotFoundError("Draft not found.")

    monkeypatch.setattr(quiz_router, "QuizDraftService", _Svc)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/quiz/drafts/d1")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_quiz_draft_programmer_error_500(monkeypatch: pytest.MonkeyPatch):
    class _Svc:
        async def get_draft(self, draft_id: str, user_id: str):
            raise ValueError("programmer bug")

    monkeypatch.setattr(quiz_router, "QuizDraftService", _Svc)
    async with AsyncClient(
        transport=ASGITransport(app=_app(), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/quiz/drafts/d1")
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_patch_validation_400(monkeypatch: pytest.MonkeyPatch):
    class _Svc:
        async def patch_draft(self, draft_id: str, user_id: str, req):
            raise QuizDraftValidationError("Draft can no longer be edited.")

    monkeypatch.setattr(quiz_router, "QuizDraftService", _Svc)
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.patch("/api/v1/quiz/drafts/d1", json={"title": "x"})
    assert response.status_code == 400
