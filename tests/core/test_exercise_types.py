from types import SimpleNamespace

from api.core.learning.exercise_gen import select_exercise_type
from api.core.learning.exercise_types import (
    FillBlankOutput,
    MatchingOutput,
    MatchingPair,
    serialize_exercise_result,
)
from api.core.learning import exercise_service as exercise_service_module
from api.core.learning.exercise_service import ExerciseService


def test_select_exercise_type_covers_new_bloom_mapping():
    assert select_exercise_type(1, 0.1) == "true_false"
    assert select_exercise_type(2, 0.5) == "fill_blank"
    assert select_exercise_type(2, 0.7) == "matching"
    assert select_exercise_type(4, 0.8) == "multi_correct"
    assert select_exercise_type(5, 0.9) == "short_answer"
    assert select_exercise_type(6, 0.2) == "mcq"


def test_evaluate_answer_handles_true_false_fill_blank_multi_correct_and_ordering():
    service = ExerciseService()
    try:
        true_false = SimpleNamespace(
            exercise_type="true_false",
            correct_answer=True,
            correct_option="True",
            concept_name="Concept",
            question="Question",
            rubric=[],
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(true_false, {"boolean": True}) == (True, "True")

        fill_blank = SimpleNamespace(
            exercise_type="fill_blank",
            correct_answer=["động năng", "dong nang"],
            correct_option="động năng",
            concept_name="Concept",
            question="Question",
            rubric=[],
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(fill_blank, {"blanks": ["Động năng"]}) == (True, "Động năng")

        multi_correct = SimpleNamespace(
            exercise_type="multi_correct",
            correct_answer=["A", "C"],
            correct_option="A, C",
            concept_name="Concept",
            question="Question",
            rubric=[],
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(multi_correct, {"choices": ["C", "A"]}) == (True, "A, C")

        ordering = SimpleNamespace(
            exercise_type="ordering",
            correct_answer=["Bước 1", "Bước 2", "Bước 3"],
            correct_option="",
            concept_name="Concept",
            question="Question",
            rubric=[],
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(ordering, {"ordering": ["Bước 1", "Bước 2", "Bước 3"]}) == (
            True,
            "Bước 1 → Bước 2 → Bước 3",
        )

        matching = SimpleNamespace(
            exercise_type="matching",
            correct_answer={"Khái niệm A": "Định nghĩa A", "Khái niệm B": "Định nghĩa B"},
            correct_option="",
            concept_name="Concept",
            question="Question",
            rubric=[],
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(
            matching,
            {"matching": {"Khái niệm A": "Định nghĩa A", "Khái niệm B": "Định nghĩa B"}},
        ) == (
            True,
            "Khái niệm A -> Định nghĩa A, Khái niệm B -> Định nghĩa B",
        )
    finally:
        service.close()


def test_serialize_exercise_result_normalizes_fill_blank_and_matching_payloads():
    fill_blank = FillBlankOutput(
        question="ignored",
        sentence="Động năng được tính bằng công thức _____.",
        blank_answers=["Wđ", "Wd"],
        hint="Liên quan đến năng lượng chuyển động",
        explanation_correct="Đúng",
        explanation_incorrect="Sai",
    )
    matching = MatchingOutput(
        question="Ghép khái niệm với định nghĩa phù hợp.",
        pairs=[
            MatchingPair(left="Vận tốc", right="Độ lớn và hướng của chuyển động"),
            MatchingPair(left="Gia tốc", right="Độ biến thiên vận tốc theo thời gian"),
            MatchingPair(left="Lực", right="Tác dụng làm vật đổi trạng thái chuyển động"),
        ],
        right_items=[
            "Độ biến thiên vận tốc theo thời gian",
            "Tác dụng làm vật đổi trạng thái chuyển động",
            "Độ lớn và hướng của chuyển động",
        ],
        explanation_correct="Đúng",
        explanation_incorrect="Sai",
    )

    fill_blank_payload = serialize_exercise_result(fill_blank)
    matching_payload = serialize_exercise_result(matching)

    assert fill_blank_payload["question"] == "Động năng được tính bằng công thức _____."
    assert fill_blank_payload["correct_answer"] == ["Wđ", "Wd"]
    assert matching_payload["right_items"][0] == "Độ biến thiên vận tốc theo thời gian"
    assert matching_payload["correct_answer"]["Vận tốc"] == "Độ lớn và hướng của chuyển động"


def test_evaluate_answer_updates_short_answer_feedback(monkeypatch):
    service = ExerciseService()
    monkeypatch.setattr(
        exercise_service_module,
        "evaluate_short_answer",
        lambda **_kwargs: {
            "is_correct": True,
            "explanation": "Đáp án đạt rubric cốt lõi.",
            "score": 9,
        },
    )

    short_answer = SimpleNamespace(
        exercise_type="short_answer",
        correct_answer="Mẫu trả lời",
        correct_option="Mẫu trả lời",
        concept_name="Concept",
        question="Question",
        rubric=["Ý chính", "Lập luận"],
        explanation_correct="",
        explanation_incorrect="",
    )

    try:
        assert service._evaluate_answer(short_answer, {"text": "Câu trả lời của học sinh"}) == (
            True,
            "Câu trả lời của học sinh",
        )
        assert short_answer.explanation_correct == "Đáp án đạt rubric cốt lõi."
        assert short_answer.explanation_incorrect == "Đáp án đạt rubric cốt lõi."
    finally:
        service.close()
