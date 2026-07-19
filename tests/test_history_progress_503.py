"""History list progress maps classified storage failures to retryable 503."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pymongo.errors import ServerSelectionTimeoutError
import pytest

from api.dependencies import get_current_user
from api.domains.learning import history_router
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(history_router, "is_admin_request", lambda *_a, **_k: True)


def _app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = limiter
    register_exception_handlers(app)
    app.include_router(history_router.router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return app


@pytest.mark.asyncio
async def test_list_subject_progress_storage_outage_returns_503(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        history_router,
        "list_recent_subject_progress",
        AsyncMock(side_effect=ServerSelectionTimeoutError("mongo down")),
    )
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/history/subjects/progress")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "service_unavailable"
    assert body["error"]["meta"]["retryable"] is True


@pytest.mark.asyncio
async def test_list_subject_progress_empty_is_200(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        history_router,
        "list_recent_subject_progress",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        history_router,
        "load_many_pipeline_jobs_for_user",
        AsyncMock(return_value={}),
    )
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/history/subjects/progress")
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 0


@pytest.mark.asyncio
async def test_list_subjects_storage_outage_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        history_router,
        "list_recent_pipeline_jobs",
        AsyncMock(side_effect=ServerSelectionTimeoutError("mongo down")),
    )
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/history/subjects")
    assert response.status_code == 503
    assert response.json()["error"]["meta"]["retryable"] is True


@pytest.mark.asyncio
async def test_list_subjects_empty_is_200(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        history_router,
        "list_recent_pipeline_jobs",
        AsyncMock(return_value=[]),
    )
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/history/subjects")
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 0


@pytest.mark.asyncio
async def test_list_subjects_programmer_error_is_500(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        history_router,
        "list_recent_pipeline_jobs",
        AsyncMock(side_effect=ValueError("bug")),
    )
    async with AsyncClient(
        transport=ASGITransport(app=_app(), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/history/subjects")
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_list_pipeline_jobs_storage_outage_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        history_router,
        "list_recent_pipeline_jobs",
        AsyncMock(side_effect=ServerSelectionTimeoutError("mongo down")),
    )
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.get("/api/v1/history/pipeline-jobs")
    assert response.status_code == 503
    assert response.json()["error"]["meta"]["retryable"] is True
