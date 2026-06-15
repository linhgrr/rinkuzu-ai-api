from api.core.learning.answer_eval import evaluate_answer, serialize_answer_for_history
from api.core.learning.exercise_types.payloads import TrueFalsePayload
from api.core.learning.session import ExerciseRecord


def _tf_record():
    return ExerciseRecord(
        exercise_id="ex1",
        concept_idx=0,
        concept_name="C",
        bloom_level=1,
        question="Đúng hay sai?",
        payload=TrueFalsePayload(statement="S", correct_answer=True),
    )


def test_evaluate_true_false_via_handler():
    assert evaluate_answer(_tf_record(), {"boolean": True}) == (True, "True")
    assert evaluate_answer(_tf_record(), {"boolean": False}) == (False, "False")


def test_serialize_answer_via_handler():
    assert serialize_answer_for_history(_tf_record(), {"boolean": True}) == "True"
