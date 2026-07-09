from types import SimpleNamespace

from api.domains.learning.exercise_types.payloads import TrueFalsePayload
from api.domains.learning.router import _resolve_exercise_options, _resolve_exercise_question


def _tf():
    return SimpleNamespace(
        exercise_id="ex1",
        question="Đánh giá phát biểu sau là đúng hay sai.",
        concept_name="Tích phân",
        bloom_level=2,
        payload=TrueFalsePayload(
            statement="Tích phân luôn là diện tích.",
            correct_answer=False,
        ),
    )


def test_tutor_question_includes_statement():
    q = _resolve_exercise_question(_tf())
    assert "Tích phân luôn là diện tích." in q  # Bug #1: statement must reach the tutor


def test_tutor_options_are_true_false():
    assert _resolve_exercise_options(_tf()) == ["True", "False"]
