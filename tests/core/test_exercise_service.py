from types import SimpleNamespace

import pytest

from api.services import exercise_service as exercise_service_module
from api.services.exercise_service import ExerciseService


def test_exercise_service_uses_separate_request_and_prefetch_timeouts(monkeypatch):
    monkeypatch.setattr(
        exercise_service_module,
        "settings",
        SimpleNamespace(
            llm_max_workers=4,
            llm_max_concurrency=2,
            llm_request_timeout_sec=90,
            llm_prefetch_timeout_sec=210,
        ),
    )

    service = ExerciseService()
    try:
        assert service._request_llm_timeout_sec == 90
        assert service._prefetch_llm_timeout_sec == 210
    finally:
        service.close()


@pytest.mark.anyio
async def test_eager_prefetch_uses_prefetch_timeout(monkeypatch):
    monkeypatch.setattr(
        exercise_service_module,
        "settings",
        SimpleNamespace(
            llm_max_workers=2,
            llm_max_concurrency=None,
            llm_request_timeout_sec=90,
            llm_prefetch_timeout_sec=210,
        ),
    )

    service = ExerciseService()
    captured: dict[str, float] = {}

    async def fake_generate_exercise_dedup(**kwargs):
        captured["timeout_sec"] = kwargs["timeout_sec"]
        return {
            "question": "Q",
            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "correct_option": "A",
            "explanation_correct": "ok",
            "explanation_incorrect": "no",
        }

    monkeypatch.setattr(service, "_generate_exercise_dedup", fake_generate_exercise_dedup)

    session = SimpleNamespace(
        env=SimpleNamespace(
            get_session_stats=lambda: {"step": 0},
            get_concept_mastery=lambda: [0.2],
        ),
        concept_map={"concept-1": 0},
        concept_names={"concept-1": "Concept 1"},
        concept_definitions={"concept-1": "Definition 1"},
        _prefetch_cache={},
    )

    try:
        await service.eager_generate_first_exercise(session)
        assert captured["timeout_sec"] == 210
        assert session._prefetch_cache["eager"]["concept_idx"] == 0
        assert session._prefetch_cache["eager"]["bloom_level"] == 1
    finally:
        service.close()
