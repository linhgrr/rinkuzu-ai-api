import asyncio

from api.domains.learning import exercise_gen
from api.domains.learning.exercise_types import (
    ExerciseOptions,
    ExerciseType,
    MCQOutput,
    ShortAnswerEvaluationOutput,
)
from api.domains.learning.prompts.grading import TheoryOutput
from api.shared import retry as retry_module


def test_generate_exercise_serializes_structured_output(monkeypatch):
    # Retry + reask live in the LLM client (see test_llm_structured_instructor
    # for the attempt-count contract). Here we only verify exercise_gen
    # serializes a structured result into the API payload shape.
    def _select_type(_bloom_level, _mastery):
        return ExerciseType.MCQ

    monkeypatch.setattr(exercise_gen, "select_exercise_type", _select_type)

    async def _fake_invoke(*, schema, messages, action, temperature=0.3):
        assert schema is MCQOutput
        assert messages
        assert action == "adaptive_exercise"
        return MCQOutput(
            question="Động năng của vật phụ thuộc vào yếu tố nào?",
            options=ExerciseOptions(
                A="Khối lượng và vận tốc",
                B="Chỉ khối lượng",
                C="Chỉ vận tốc",
                D="Nhiệt độ",
            ),
            correct_option="A",
            explanation_correct="Đúng vì công thức là 1/2mv^2.",
            explanation_incorrect="Sai vì động năng phụ thuộc cả khối lượng và vận tốc.",
        )

    monkeypatch.setattr(exercise_gen, "_invoke_structured_llm", _fake_invoke)

    result = asyncio.run(
        exercise_gen.generate_exercise(
            concept_name="Động năng",
            concept_definition="Động năng là năng lượng mà vật có do chuyển động.",
            bloom_level=3,
        )
    )

    assert result is not None
    assert result["exercise_type"] == ExerciseType.MCQ
    assert result["payload"]["correct_option"] == "A"


def test_evaluate_short_answer_returns_model_dump(monkeypatch):
    async def _graded_output(**_kwargs):
        return ShortAnswerEvaluationOutput(is_correct=True, explanation="Đủ ý.", score=9)

    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (1, 0.0))
    monkeypatch.setattr(exercise_gen, "_invoke_structured_llm", _graded_output)

    result = asyncio.run(
        exercise_gen.evaluate_short_answer(
            concept_name="Quán tính",
            question="Quán tính là gì?",
            rubric=["Nêu được khái niệm", "Có ví dụ ngắn"],
            sample_answer="Quán tính là xu hướng giữ nguyên trạng thái chuyển động.",
            student_answer="Là xu hướng giữ nguyên trạng thái.",
        )
    )

    assert result == {"is_correct": True, "explanation": "Đủ ý.", "score": 9}


def test_generate_theory_returns_fallback_after_retries(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (2, 0.0))

    async def _always_fail(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(exercise_gen, "_invoke_structured_llm", _always_fail)

    result = asyncio.run(
        exercise_gen.generate_theory(
            concept_name="Động lượng",
            concept_definition="Động lượng là đại lượng đặc trưng cho chuyển động.",
            bloom_level=2,
        )
    )

    assert result == {
        "content": "Lý thuyết cơ bản về Động lượng: Động lượng là đại lượng đặc trưng cho chuyển động.",
        "examples": ["Ví dụ 1: ...", "Ví dụ 2: ..."],
    }


def test_generate_theory_returns_model_dump_on_success(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (1, 0.0))

    async def _fake_invoke(**_kwargs):
        return TheoryOutput(content="Nội dung", examples=["Ví dụ 1"])

    monkeypatch.setattr(exercise_gen, "_invoke_structured_llm", _fake_invoke)

    result = asyncio.run(
        exercise_gen.generate_theory(
            concept_name="Động lượng",
            concept_definition="Động lượng là đại lượng đặc trưng cho chuyển động.",
            bloom_level=2,
        )
    )

    assert result == {
        "content": "Nội dung",
        "examples": ["Ví dụ 1"],
    }
