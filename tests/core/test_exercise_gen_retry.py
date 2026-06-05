from types import SimpleNamespace

from pydantic import BaseModel

from api.core.shared import llm as llm_module
from api.core.shared.llm import (
    _resolve_shared_llm_model,
    invoke_structured_completion,
    resolve_retry_policy,
)


class OutputSchema(BaseModel):
    message: str


def test_resolve_retry_policy_uses_backend_settings(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=5,
            llm_retry_backoff_sec=2.5,
            exercise_llm_model=None,
            llm_model="shared-model",
        ),
    )

    assert resolve_retry_policy() == (5, 2.5)


def test_resolve_exercise_llm_model_prefers_exercise_specific_override(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=3,
            llm_retry_backoff_sec=1.0,
            exercise_llm_model="exercise-model",
            llm_model="shared-model",
        ),
    )

    assert _resolve_shared_llm_model("shared-model") == "exercise-model"


def test_resolve_exercise_llm_model_falls_back_to_shared_llm_model(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=3,
            llm_retry_backoff_sec=1.0,
            exercise_llm_model=None,
            llm_model="shared-model",
        ),
    )

    assert _resolve_shared_llm_model(None) == "shared-model"


def test_invoke_structured_completion_uses_json_object_response_format(monkeypatch):
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"message": "ok"}'))]
        )

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="https://api.deepseek.com",
            llm_api_key="test-key",  # pragma: allowlist secret
            llm_model="model-x",
            llm_timeout_sec=30,
            exercise_llm_model=None,
            llm_retry_attempts=1,
            llm_retry_backoff_sec=0.0,
        ),
    )

    result = invoke_structured_completion(
        schema=OutputSchema,
        model="model-x",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
    )

    assert result == OutputSchema(message="ok")
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.2
    assert captured["model"] == "model-x"
    assert captured["num_retries"] == 0
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert "Return valid json only" in messages[0]["content"]
