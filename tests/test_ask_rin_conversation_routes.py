from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_current_user
import api.domains.assistant.router as assistant_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant_router.router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("encoded_context_id", "decoded_context_id"),
    [
        ("quiz%3A905f74bcce2d0a5aba72bf7d68f9bba7", "quiz:905f74bcce2d0a5aba72bf7d68f9bba7"),
        ("adaptive%3Asession-1%3Aexercise_2", "adaptive:session-1:exercise_2"),
    ],
)
def test_read_conversation_accepts_namespaced_context_ids(
    monkeypatch: pytest.MonkeyPatch,
    encoded_context_id: str,
    decoded_context_id: str,
) -> None:
    get_conversation = AsyncMock(return_value=None)
    monkeypatch.setattr(assistant_router, "get_conversation", get_conversation)

    response = _client().get(f"/api/v1/ask-rin-chan/conversations/{encoded_context_id}")

    assert response.status_code == 200
    get_conversation.assert_awaited_once_with("user-1", decoded_context_id)


def test_read_conversation_still_rejects_unsafe_context_ids() -> None:
    response = _client().get("/api/v1/ask-rin-chan/conversations/quiz%3Aunsafe.context")

    assert response.status_code == 422
