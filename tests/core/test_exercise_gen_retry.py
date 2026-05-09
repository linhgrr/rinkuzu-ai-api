from types import SimpleNamespace

from pydantic import BaseModel

from api.core.shared import llm as llm_module
from api.core.shared.llm import _resolve_shared_llm_model, get_structured_llm, resolve_retry_policy


def test_resolve_retry_policy_uses_backend_settings(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=5,
            llm_retry_backoff_sec=2.5,
            exercise_llm_model=None,
            openai_model="shared-model",
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
            openai_model="shared-model",
        ),
    )

    assert _resolve_shared_llm_model("shared-model") == "exercise-model"


def test_resolve_exercise_llm_model_falls_back_to_openai_model(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=3,
            llm_retry_backoff_sec=1.0,
            exercise_llm_model=None,
            openai_model="shared-model",
        ),
    )

    assert _resolve_shared_llm_model(None) == "shared-model"


def test_get_structured_llm_uses_provider_native_json_schema(monkeypatch):
    captured: dict[str, object] = {}

    class OutputSchema(BaseModel):
        message: str

    class FakeChatModel:
        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured.update(kwargs)
            return "structured-runnable"

    def _fake_get_llm(**_kwargs):
        return FakeChatModel()

    monkeypatch.setattr(llm_module, "get_llm", _fake_get_llm)

    result = get_structured_llm(OutputSchema, temperature=0.2)

    assert result == "structured-runnable"
    assert captured["schema"] is OutputSchema
    assert captured["method"] == "json_schema"
    assert captured["strict"] is True
