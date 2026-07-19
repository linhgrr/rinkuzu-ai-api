"""Minimal 200/404/503 coverage for quiz draft delete."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from api.dependencies import get_current_user
from api.domains.quiz import router as quiz_router
from api.domains.quiz.draft_service import QuizDraftNotFoundError, QuizDraftService
from api.exceptions import AppError, register_exception_handlers
from api.rate_limit import limiter


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr(quiz_router, "is_admin_request", lambda *a, **k: True)


def _app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = limiter
    register_exception_handlers(app)
    app.include_router(quiz_router.drafts_router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return app


def _draft(**overrides):
    base = {
        "draft_id": "draft-1",
        "user_id": "user-1",
        "title": "t",
        "description": "",
        "category_id": None,
        "prompt": None,
        "pdf": {
            "s3_key": "uploads/quiz_extract/user-1/file.pdf",
            "file_name": "file.pdf",
            "file_size": 12,
            "page_count": 1,
        },
        "status": "cancelled",
        "progress": {"processed": 0, "total": 3, "percent": 0},
        "questions": [],
        "error": None,
        "submitted_quiz_id": None,
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
        "expires_at": "2020-01-03T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_delete_route_returns_200(monkeypatch):
    service = QuizDraftService()
    monkeypatch.setattr(
        "api.domains.quiz.router.QuizDraftService",
        lambda: service,
    )
    monkeypatch.setattr(service, "delete_draft", AsyncMock(return_value=_draft()))

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.delete("/api/v1/quiz/drafts/draft-1")

    assert response.status_code == 200
    assert response.json()["data"]["draft"]["draft_id"] == "draft-1"


@pytest.mark.asyncio
async def test_delete_route_returns_404_when_absent(monkeypatch):
    service = QuizDraftService()
    monkeypatch.setattr("api.domains.quiz.router.QuizDraftService", lambda: service)
    monkeypatch.setattr(
        service,
        "delete_draft",
        AsyncMock(side_effect=QuizDraftNotFoundError("Draft not found.")),
    )

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.delete("/api/v1/quiz/drafts/draft-1")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_route_returns_503_on_app_error(monkeypatch):
    service = QuizDraftService()
    monkeypatch.setattr("api.domains.quiz.router.QuizDraftService", lambda: service)
    monkeypatch.setattr(
        service,
        "delete_draft",
        AsyncMock(
            side_effect=AppError(
                code="service_unavailable",
                message="Draft cleanup failed",
                detail="mongo unavailable",
                status_code=503,
                meta={"retryable": True, "draft_id": "draft-1"},
            )
        ),
    )

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        response = await client.delete("/api/v1/quiz/drafts/draft-1")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


@pytest.mark.asyncio
async def test_delete_route_returns_500_on_unexpected_error(monkeypatch):
    """Non-AppError programmer/unexpected errors re-raise to global 500 handler."""
    service = QuizDraftService()
    monkeypatch.setattr("api.domains.quiz.router.QuizDraftService", lambda: service)
    monkeypatch.setattr(
        service,
        "delete_draft",
        AsyncMock(side_effect=RuntimeError("unexpected programmer error")),
    )

    async with AsyncClient(
        transport=ASGITransport(app=_app(), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.delete("/api/v1/quiz/drafts/draft-1")

    assert response.status_code == 500
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "internal_error"
    assert payload["error"]["message"] == "Internal server error"
    # Global handler must not leak the raw programmer exception string.
    assert "unexpected programmer error" not in response.text


@pytest.mark.asyncio
async def test_delete_service_cancel_is_monotonic(monkeypatch):
    """Producer cancel runs before metadata delete; concurrent absence converges to 200."""
    service = QuizDraftService()
    observed = _draft(status="processing")
    cancelled = _draft(status="cancelled")
    order: list[str] = []

    async def load_once(draft_id: str, user_id: str):
        del draft_id, user_id
        order.append("get")
        return observed

    async def cancel_once(draft_id: str, user_id: str):
        del draft_id, user_id
        order.append("cancel")
        return cancelled

    async def task_cancel(draft_id: str):
        order.append(f"task:{draft_id}")

    async def delete_once(draft_id: str, user_id: str):
        del draft_id, user_id
        order.append("delete")
        return cancelled

    monkeypatch.setattr("api.domains.quiz.draft_service.load_quiz_draft_for_user", load_once)
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.request_cancel_quiz_draft_for_user",
        cancel_once,
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_tasks.quiz_draft_task_manager.cancel",
        task_cancel,
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.delete_quiz_draft_for_user",
        delete_once,
    )
    monkeypatch.setattr(
        service,
        "_delete_pdf_best_effort",
        lambda key: order.append(f"pdf:{key}"),
    )

    result = await service.delete_draft("draft-1", "user-1")
    assert result["status"] == "cancelled"
    assert order[:4] == ["get", "cancel", "task:draft-1", "delete"]
    assert order[4].startswith("pdf:")


@pytest.mark.asyncio
async def test_delete_service_absent_is_not_found(monkeypatch):
    service = QuizDraftService()

    async def load_none(draft_id: str, user_id: str):
        del draft_id, user_id

    monkeypatch.setattr("api.domains.quiz.draft_service.load_quiz_draft_for_user", load_none)

    with pytest.raises(QuizDraftNotFoundError):
        await service.delete_draft("draft-1", "user-1")


@pytest.mark.asyncio
async def test_delete_service_load_error_is_503(monkeypatch):
    from pymongo.errors import ServerSelectionTimeoutError

    service = QuizDraftService()

    async def load_fail(draft_id: str, user_id: str):
        del draft_id, user_id
        raise ServerSelectionTimeoutError("mongo down")

    monkeypatch.setattr("api.domains.quiz.draft_service.load_quiz_draft_for_user", load_fail)

    with pytest.raises(AppError) as exc_info:
        await service.delete_draft("draft-1", "user-1")
    assert exc_info.value.status_code == 503
    assert exc_info.value.meta["retryable"] is True


@pytest.mark.asyncio
async def test_delete_service_programming_error_is_not_503(monkeypatch):
    """Unexpected programming/invariant errors must not be remapped to 503."""
    service = QuizDraftService()

    async def load_fail(draft_id: str, user_id: str):
        del draft_id, user_id
        raise ValueError("invariant broken")

    monkeypatch.setattr("api.domains.quiz.draft_service.load_quiz_draft_for_user", load_fail)

    with pytest.raises(ValueError, match="invariant broken"):
        await service.delete_draft("draft-1", "user-1")


@pytest.mark.asyncio
async def test_delete_cancel_none_after_observe_converges_200(monkeypatch):
    """Cancel returning None after observation is concurrent convergence → 200."""
    service = QuizDraftService()
    observed = _draft(status="processing")

    async def load_once(draft_id: str, user_id: str):
        del draft_id, user_id
        return observed

    async def cancel_none(draft_id: str, user_id: str):
        del draft_id, user_id

    async def task_cancel(draft_id: str):
        del draft_id

    async def delete_none(draft_id: str, user_id: str):
        del draft_id, user_id

    monkeypatch.setattr("api.domains.quiz.draft_service.load_quiz_draft_for_user", load_once)
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.request_cancel_quiz_draft_for_user",
        cancel_none,
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_tasks.quiz_draft_task_manager.cancel",
        task_cancel,
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.delete_quiz_draft_for_user",
        delete_none,
    )

    result = await service.delete_draft("draft-1", "user-1")
    assert result is observed


@pytest.mark.asyncio
async def test_delete_pdf_cleanup_failure_still_200(monkeypatch):
    """Blob cleanup scheduling/function failure must not turn success into 500."""
    service = QuizDraftService()
    cancelled = _draft(status="cancelled")

    async def load_once(draft_id: str, user_id: str):
        del draft_id, user_id
        return cancelled

    async def cancel_once(draft_id: str, user_id: str):
        del draft_id, user_id
        return cancelled

    async def task_cancel(draft_id: str):
        del draft_id

    async def delete_once(draft_id: str, user_id: str):
        del draft_id, user_id
        return cancelled

    async def boom_to_thread(fn, *args, **kwargs):
        del fn, args, kwargs
        raise RuntimeError("thread pool closed")

    monkeypatch.setattr("api.domains.quiz.draft_service.load_quiz_draft_for_user", load_once)
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.request_cancel_quiz_draft_for_user",
        cancel_once,
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_tasks.quiz_draft_task_manager.cancel",
        task_cancel,
    )
    monkeypatch.setattr(
        "api.domains.quiz.draft_service.delete_quiz_draft_for_user",
        delete_once,
    )
    monkeypatch.setattr("api.domains.quiz.draft_service.asyncio.to_thread", boom_to_thread)

    result = await service.delete_draft("draft-1", "user-1")
    assert result["draft_id"] == "draft-1"
