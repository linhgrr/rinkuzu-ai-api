from types import SimpleNamespace

from api.core.shared import llm as llm_module
from api.core.shared.llm import _resolve_shared_llm_model, resolve_retry_policy


def test_resolve_retry_policy_uses_backend_settings(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=5,
            llm_retry_backoff_sec=2.5,
            exercise_llm_model=None,
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
        ),
    )

    assert _resolve_shared_llm_model("shared-model") == "exercise-model"
