"""404 / 503 / 500 semantics for DELETE /api/v1/history/subjects/{job_id}."""

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
async def test_delete_subject_returns_404_when_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        history_router,
        "delete_pipeline_job_for_user",
        AsyncMock(return_value={"deleted_job": 0, "deleted_sessions": 0}),
    )

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.delete("/api/v1/history/subjects/job-missing")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "pipeline_not_found"


@pytest.mark.asyncio
async def test_delete_subject_returns_503_on_classified_db_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        history_router,
        "delete_pipeline_job_for_user",
        AsyncMock(side_effect=ServerSelectionTimeoutError("mongo down")),
    )

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.delete("/api/v1/history/subjects/job-1")

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "service_unavailable"
    assert body["error"]["meta"]["retryable"] is True


@pytest.mark.asyncio
async def test_delete_subject_returns_503_when_mongo_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        history_router,
        "delete_pipeline_job_for_user",
        AsyncMock(side_effect=RuntimeError("MongoDB client not initialized")),
    )

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.delete("/api/v1/history/subjects/job-1")

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "service_unavailable"
    assert body["error"]["meta"]["retryable"] is True


@pytest.mark.asyncio
async def test_delete_subject_returns_500_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        history_router,
        "delete_pipeline_job_for_user",
        AsyncMock(side_effect=ValueError("unexpected programmer error")),
    )

    # raise_app_exceptions=False so the registered Exception→500 handler is observable.
    transport = ASGITransport(app=_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/api/v1/history/subjects/job-1")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "unexpected programmer error" not in response.text


@pytest.mark.asyncio
async def test_delete_subject_does_not_misclassify_runtime_error_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        history_router,
        "delete_pipeline_job_for_user",
        AsyncMock(side_effect=RuntimeError("MongoDB invariant broken")),
    )

    transport = ASGITransport(app=_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/api/v1/history/subjects/job-1")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "MongoDB invariant broken" not in response.text


@pytest.mark.asyncio
async def test_delete_pipeline_job_db_error_propagates_not_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    """Store must not collapse infra failures into deleted_job=0 (false 404)."""
    from api.shared.persistence import pipeline_jobs as store

    class _BoomSession:
        async def __aenter__(self):
            raise ServerSelectionTimeoutError("mongo down")

        async def __aexit__(self, *args):
            return False

    def _start_session() -> _BoomSession:
        return _BoomSession()

    monkeypatch.setattr(store.mongo_store, "start_session", _start_session)

    with pytest.raises(ServerSelectionTimeoutError):
        await store.delete_pipeline_job_for_user("job-1", "user-1")
