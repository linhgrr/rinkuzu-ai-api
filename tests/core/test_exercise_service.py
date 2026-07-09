from types import SimpleNamespace

import pytest

from api.domains.learning import exercise_service as exercise_service_module
from api.domains.learning.exercise_service import ExerciseService


def test_exercise_service_uses_separate_request_and_prefetch_timeouts(monkeypatch):
    monkeypatch.setattr(
        exercise_service_module,
        "settings",
        SimpleNamespace(
            llm_max_workers=4,
            llm_max_concurrency=2,
            llm_request_timeout_sec=90,
            llm_prefetch_timeout_sec=210,
            adaptive_exercise_recent_same_concept_limit=5,
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
            adaptive_exercise_recent_same_concept_limit=5,
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
            "payload": {
                "exercise_type": "mcq",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_option": "A",
            },
        }

    monkeypatch.setattr(service, "_generate_exercise_dedup", fake_generate_exercise_dedup)

    session = SimpleNamespace(
        env=SimpleNamespace(
            get_session_stats=lambda: {"step": 0},
            get_concept_mastery=lambda: [0.2],
        ),
        q_net=None,
        current_obs=None,
        device=None,
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


def test_get_recent_same_concept_exercises_respects_setting_and_order(monkeypatch):
    monkeypatch.setattr(
        exercise_service_module,
        "settings",
        SimpleNamespace(
            llm_max_workers=2,
            llm_max_concurrency=None,
            llm_request_timeout_sec=90,
            llm_prefetch_timeout_sec=210,
            adaptive_exercise_recent_same_concept_limit=2,
        ),
    )

    service = ExerciseService()
    session = SimpleNamespace(
        exercise_history=[
            SimpleNamespace(
                concept_idx=0,
                question="Q1",
                exercise_type="mcq",
                bloom_level=1,
                statement=None,
                hint=None,
                options={"A": "1"},
                items=[],
                pairs=[],
                right_items=[],
                rubric=[],
                correct_option="A",
                correct_answer=None,
            ),
            SimpleNamespace(
                concept_idx=1,
                question="other concept",
                exercise_type="mcq",
                bloom_level=2,
                statement=None,
                hint=None,
                options={"A": "x"},
                items=[],
                pairs=[],
                right_items=[],
                rubric=[],
                correct_option="A",
                correct_answer=None,
            ),
            SimpleNamespace(
                concept_idx=0,
                question="Q2",
                exercise_type="fill_blank",
                bloom_level=3,
                statement=None,
                hint="hint",
                options={},
                items=[],
                pairs=[],
                right_items=[],
                rubric=[],
                correct_option="",
                correct_answer=["ans"],
            ),
            SimpleNamespace(
                concept_idx=0,
                question="Q3",
                exercise_type="true_false",
                bloom_level=2,
                statement="S3",
                hint=None,
                options={},
                items=[],
                pairs=[],
                right_items=[],
                rubric=[],
                correct_option="",
                correct_answer=True,
            ),
        ]
    )

    try:
        recent = service._get_recent_same_concept_exercises(session, concept_idx=0)
        assert [item["question"] for item in recent] == ["Q3", "Q2"]
        assert all(item["question"] != "other concept" for item in recent)
    finally:
        service.close()
