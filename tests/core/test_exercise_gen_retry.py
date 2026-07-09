from types import SimpleNamespace

import api.config as config_module
from api.shared import llm as llm_module
from api.shared.llm import _resolve_shared_llm_model
from api.shared.retry import resolve_llm_retry_policy


def test_resolve_llm_retry_policy_uses_backend_settings(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_retry_attempts=5,
            llm_retry_backoff_sec=2.5,
        ),
    )

    assert resolve_llm_retry_policy() == (5, 2.5)


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
