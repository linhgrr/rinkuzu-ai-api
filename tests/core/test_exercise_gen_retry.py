from types import SimpleNamespace

from api.core import exercise_gen
from api.core.exercise_gen import _resolve_exercise_llm_model, _resolve_retry_policy


def test_resolve_retry_policy_uses_backend_settings(monkeypatch):
    monkeypatch.setattr(
        exercise_gen,
        "get_settings",
        lambda: SimpleNamespace(
            adaptive_llm_retry_attempts=5,
            adaptive_llm_retry_backoff_sec=2.5,
            adaptive_exercise_llm_model=None,
        ),
    )

    assert _resolve_retry_policy() == (5, 2.5)


def test_resolve_exercise_llm_model_prefers_exercise_specific_override(monkeypatch):
    monkeypatch.setattr(
        exercise_gen,
        "get_settings",
        lambda: SimpleNamespace(
            adaptive_llm_retry_attempts=3,
            adaptive_llm_retry_backoff_sec=1.0,
            adaptive_exercise_llm_model="exercise-model",
        ),
    )

    assert _resolve_exercise_llm_model("shared-model") == "exercise-model"
