from types import SimpleNamespace

from api.core.exercise_gen import select_exercise_type
from api.services.exercise_service import ExerciseService


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
