from types import SimpleNamespace
from unittest.mock import patch

from api.core.learning import exercise_service as exercise_service_module
from api.core.learning import exercise_types as exercise_types_module
from api.core.learning.exercise_service import ExerciseService
from api.core.learning.exercise_types import (
    ExerciseType,
    FillBlankOutput,
    MatchingOutput,
    MatchingPair,
    OrderingOutput,
    select_exercise_type,
    serialize_exercise_result,
    shuffle_ordering_items,
)


def test_select_exercise_type_covers_new_bloom_mapping():
    # Mock random.choices to return a deterministic value
    with patch.object(exercise_types_module._rng, "choices") as mock_choices:
        mock_choices.return_value = [ExerciseType.TRUE_FALSE]
        assert select_exercise_type(1, 0.1) == ExerciseType.TRUE_FALSE

        mock_choices.return_value = [ExerciseType.FILL_BLANK]
        assert select_exercise_type(2, 0.5) == ExerciseType.FILL_BLANK


def test_select_exercise_type_uses_correct_weights_for_mastery():
    with patch.object(exercise_types_module._rng, "choices") as mock_choices:
        mock_choices.return_value = [ExerciseType.MCQ]

        # Test low mastery (< 0.4)
        select_exercise_type(1, 0.1)
        # For bloom 1, low mastery weights are: TRUE_FALSE: 70, MCQ: 30
        _args, kwargs = mock_choices.call_args
        assert kwargs["weights"] == [70, 30]

        # Test mid mastery (0.4 - 0.7)
        select_exercise_type(2, 0.5)
        # For bloom 2, mid mastery weights are: TRUE_FALSE: 20, MCQ: 40, FILL_BLANK: 30, MATCHING: 10
        _args, kwargs = mock_choices.call_args
        assert kwargs["weights"] == [20, 40, 30, 10]

        # Test high mastery (>= 0.7)
        select_exercise_type(3, 0.8)
        # For bloom 3, high mastery weights are: MCQ: 5, FILL_BLANK: 20, MATCHING: 20, MULTI_CORRECT: 35, ORDERING: 20
        _args, kwargs = mock_choices.call_args
        assert kwargs["weights"] == [5, 20, 20, 35, 20]


def test_evaluate_answer_handles_true_false_fill_blank_multi_correct_and_ordering():
    from api.core.learning.exercise_types.payloads import (
        FillBlankPayload,
        MatchingPayload,
        MultiCorrectPayload,
        OrderingPayload,
        TrueFalsePayload,
    )

    service = ExerciseService()
    try:
        true_false = SimpleNamespace(
            payload=TrueFalsePayload(statement="S", correct_answer=True),
            concept_name="Concept",
            question="Question",
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(true_false, {"boolean": True}) == (True, "True")

        fill_blank = SimpleNamespace(
            payload=FillBlankPayload(
                sentence="S", hint="H", blank_answers=["động năng", "dong nang"]
            ),
            concept_name="Concept",
            question="Question",
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(fill_blank, {"blanks": ["Động năng"]}) == (
            True,
            "Động năng",
        )

        multi_correct = SimpleNamespace(
            payload=MultiCorrectPayload(
                options={"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"},
                correct_options=["A", "C"],
            ),
            concept_name="Concept",
            question="Question",
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(multi_correct, {"choices": ["C", "A"]}) == (True, "A, C")

        ordering = SimpleNamespace(
            payload=OrderingPayload(correct_order=["Bước 1", "Bước 2", "Bước 3"]),
            concept_name="Concept",
            question="Question",
            explanation_correct="",
            explanation_incorrect="",
        )
        assert service._evaluate_answer(ordering, {"ordering": ["Bước 1", "Bước 2", "Bước 3"]}) == (
            True,
            "Bước 1 → Bước 2 → Bước 3",
        )

        matching = SimpleNamespace(
            payload=MatchingPayload(
                pairs=[
                    {"left": "Khái niệm A", "right": "Định nghĩa A"},
                    {"left": "Khái niệm B", "right": "Định nghĩa B"},
                ]
            ),
            concept_name="Concept",
            question="Question",
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
        question="Hãy điền từ thích hợp vào chỗ trống.",
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
        explanation_correct="Đúng",
        explanation_incorrect="Sai",
    )

    fill_blank_payload = serialize_exercise_result(fill_blank)
    matching_payload = serialize_exercise_result(matching)

    assert fill_blank_payload["question"] == "Hãy điền từ thích hợp vào chỗ trống."
    assert fill_blank_payload["sentence"] == "Động năng được tính bằng công thức _____."
    assert fill_blank_payload["correct_answer"] == ["Wđ", "Wd"]
    assert sorted(matching_payload["right_items"]) == sorted(
        [
            "Độ lớn và hướng của chuyển động",
            "Độ biến thiên vận tốc theo thời gian",
            "Tác dụng làm vật đổi trạng thái chuyển động",
        ]
    )
    assert matching_payload["correct_answer"]["Vận tốc"] == "Độ lớn và hướng của chuyển động"


def test_shuffle_ordering_items_uses_guard_when_shuffle_keeps_original_order():
    original = ["Bước 1", "Bước 2", "Bước 3"]

    with patch.object(exercise_types_module._rng, "shuffle", side_effect=lambda _items: None):
        shuffled = shuffle_ordering_items(original)

    assert shuffled == ["Bước 2", "Bước 3", "Bước 1"]
    assert shuffled != original


def test_serialize_exercise_result_normalizes_ordering_payload_from_correct_order():
    ordering = OrderingOutput(
        question="Sắp xếp các bước theo đúng trình tự.",
        items=["Sai 1", "Sai 2", "Sai 3"],
        correct_order=["Bước 1", "Bước 2", "Bước 3"],
        explanation_correct="Đúng",
        explanation_incorrect="Sai",
    )

    with patch.object(
        exercise_types_module._rng, "shuffle", side_effect=lambda items: items.reverse()
    ):
        ordering_payload = serialize_exercise_result(ordering)

    assert ordering_payload["correct_answer"] == ["Bước 1", "Bước 2", "Bước 3"]
    assert ordering_payload["items"] == ["Bước 3", "Bước 2", "Bước 1"]
    assert ordering_payload["items"] != ordering_payload["correct_answer"]


def test_evaluate_answer_updates_short_answer_feedback(monkeypatch):
    from api.core.learning.exercise_types.payloads import ShortAnswerPayload

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
        payload=ShortAnswerPayload(rubric=["Ý chính", "Lập luận"], sample_answer="Mẫu trả lời"),
        concept_name="Concept",
        question="Question",
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
