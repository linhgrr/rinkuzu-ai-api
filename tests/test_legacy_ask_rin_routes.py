from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_chunk_chroma_store, get_current_user, get_session_manager
from api.domains.assistant import legacy_router
from api.rate_limit import limiter


class _FakeAskRinService:
    def __init__(self) -> None:
        self.generate_response = AsyncMock(return_value="Shared service answer")


def _client() -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(legacy_router.legacy_quiz_router)
    app.include_router(legacy_router.legacy_session_router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    app.dependency_overrides[get_session_manager] = object
    app.dependency_overrides[get_chunk_chroma_store] = object
    return TestClient(app, raise_server_exceptions=False)


def test_legacy_quiz_route_delegates_to_shared_ask_rin_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeAskRinService()
    monkeypatch.setattr(legacy_router, "get_ask_rin_chan_service", lambda: service)

    response = _client().post(
        "/api/v1/quiz/ask-ai",
        json={
            "question": "What is 2 + 2?",
            "options": ["3", "4"],
            "userQuestion": "Please explain",
            "chatHistory": [{"role": "assistant", "content": "Let us reason."}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["explanation"] == "Shared service answer"
    context = service.generate_response.await_args.args[0]
    assert context.action == "ask_rin_chan"
    assert context.user_question == "Please explain"
    assert context.chat_history == [{"role": "assistant", "content": "Let us reason."}]


def test_legacy_adaptive_route_resolves_server_owned_exercise_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeAskRinService()
    exercise = SimpleNamespace(exercise_id="exercise-1", concept_name="Addition", bloom_level=2)
    session = SimpleNamespace(current_exercise=exercise, exercise_history=[])
    monkeypatch.setattr(legacy_router, "get_ask_rin_chan_service", lambda: service)
    monkeypatch.setattr(legacy_router, "resolve_user_session", AsyncMock(return_value=session))
    monkeypatch.setattr(legacy_router, "_resolve_exercise_question", lambda _: "What is 2 + 2?")
    monkeypatch.setattr(legacy_router, "_resolve_exercise_options", lambda _: ["3", "4"])
    monkeypatch.setattr(
        legacy_router,
        "_build_rag_context",
        AsyncMock(return_value="Relevant course context"),
    )

    response = _client().post(
        "/api/v1/session/session-1/chat",
        json={"userQuestion": "Why is it four?", "stream": False},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {"explanation": "Shared service answer"}
    context = service.generate_response.await_args.args[0]
    assert context.question == "What is 2 + 2?"
    assert context.rag_context == "Relevant course context"


def test_legacy_routes_are_explicitly_deprecated_in_openapi() -> None:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(legacy_router.legacy_quiz_router)
    app.include_router(legacy_router.legacy_session_router)

    paths = app.openapi()["paths"]
    assert paths["/api/v1/quiz/ask-ai"]["post"]["deprecated"] is True
    assert paths["/api/v1/session/{session_id}/chat"]["post"]["deprecated"] is True
