import asyncio
from types import SimpleNamespace

import pytest

from api.domains.learning import exercise_service as exercise_service_module
from api.domains.learning.exercise_service import ExerciseService
from api.domains.learning.exercise_types.payloads import MCQPayload
from api.domains.learning.session import ExerciseRecord
from api.exceptions import AppError
from api.shared.llm_usage import current_user_id


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
        id_to_concept={0: "concept-1"},
        _prefetch_cache={},
    )

    try:
        await service.eager_generate_first_exercise(session)
        assert captured["timeout_sec"] == 210
        assert session._prefetch_cache["eager"]["concept_idx"] == 0
        assert session._prefetch_cache["eager"]["bloom_level"] == 1
    finally:
        service.close()


@pytest.mark.anyio
async def test_generate_exercise_dedup_sets_usage_user_from_session(monkeypatch):
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
    captured: dict[str, str | None] = {}

    async def fake_generate_exercise(*_args):
        captured["user_id"] = current_user_id.get()
        return {"question": "Q"}

    monkeypatch.setattr(exercise_service_module, "generate_exercise", fake_generate_exercise)

    service = ExerciseService()
    session = SimpleNamespace(
        session_id="session-1",
        user_id="user-1",
        exercise_history=[],
    )

    try:
        result = await service._generate_exercise_dedup(
            session=session,
            concept_idx=0,
            bloom_level=1,
            concept_name="Concept",
            concept_def="Definition",
            mastery=0.2,
        )
    finally:
        service.close()

    assert result == {"question": "Q"}
    assert captured["user_id"] == "user-1"
    assert current_user_id.get() is None


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


class _SubmitEnv:
    def __init__(self):
        self.step_count = 0

    def step(self, action_id, *, human_correct):
        del action_id, human_correct
        self.step_count += 1
        return None, 0.0, False, False, {"step": self.step_count}

    def get_concept_mastery(self):
        return [0.6]


class _PersistingManager:
    def __init__(self):
        self.persist_calls = 0

    async def persist_subject_progress(self, session):
        del session
        self.persist_calls += 1
        return True

    def remove_session(self, session_id):
        del session_id
        raise AssertionError("session should not be removed when persistence succeeds")


def _submit_session():
    return SimpleNamespace(
        session_id="session-1",
        _lock=asyncio.Lock(),
        current_exercise=ExerciseRecord(
            exercise_id="exercise-1",
            concept_idx=0,
            concept_name="Concept",
            bloom_level=1,
            question="Question?",
            payload=MCQPayload(options={"A": "Correct", "B": "Wrong"}, correct_option="A"),
        ),
        exercise_history=[],
        total_correct=0,
        total_answered=0,
        _pending_action=0,
        env=_SubmitEnv(),
        status="active",
        submission_receipts={},
    )


@pytest.mark.anyio
async def test_submit_answer_is_idempotent_for_same_exercise_and_key():
    manager = _PersistingManager()
    service = ExerciseService(session_manager=manager)
    session = _submit_session()

    first = await service.submit_answer(
        session,
        {"choice": "A"},
        exercise_id="exercise-1",
        idempotency_key="submit-1",
    )
    second = await service.submit_answer(
        session,
        {"choice": "A"},
        exercise_id="exercise-1",
        idempotency_key="submit-1",
    )

    assert first == second
    assert session.total_answered == 1
    assert session.env.step_count == 1
    assert manager.persist_calls == 1


@pytest.mark.anyio
async def test_submit_answer_rejects_stale_exercise_id():
    service = ExerciseService(session_manager=_PersistingManager())
    session = _submit_session()

    with pytest.raises(AppError) as exc_info:
        await service.submit_answer(
            session,
            {"choice": "A"},
            exercise_id="other-exercise",
            idempotency_key="submit-1",
        )

    assert exc_info.value.status_code == 409
    assert session.total_answered == 0
