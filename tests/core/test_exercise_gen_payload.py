import asyncio
from unittest.mock import patch

from api.domains.learning import exercise_gen
from api.domains.learning.exercise_types import ExerciseType, TrueFalseOutput


def test_generate_exercise_includes_nested_payload(monkeypatch):
    monkeypatch.setattr(
        exercise_gen, "select_exercise_type", lambda *_a, **_k: ExerciseType.TRUE_FALSE
    )

    fake = TrueFalseOutput(
        question="Đúng hay sai?",
        statement="Số 2 là số nguyên tố.",
        correct_answer=True,
        explanation_correct="Đúng",
        explanation_incorrect="Sai",
    )

    async def _fake_invoke(**_kwargs):
        return fake

    with patch.object(exercise_gen, "_invoke_structured_llm", _fake_invoke):
        data = asyncio.run(exercise_gen.generate_exercise("Số nguyên tố", "def", 1))

    assert data is not None
    assert data["payload"] == {
        "exercise_type": "true_false",
        "statement": "Số 2 là số nguyên tố.",
        "correct_answer": True,
    }
