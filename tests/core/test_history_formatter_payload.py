import json
from types import SimpleNamespace

from api.domains.learning.exercise_types.payloads import TrueFalsePayload
from api.domains.learning.history_formatter import format_exercise_history


def test_formatter_includes_statement_from_payload():
    rec = SimpleNamespace(
        exercise_id="ex1",
        question="Đúng hay sai?",
        concept_idx=0,
        concept_name="C",
        bloom_level=1,
        payload=TrueFalsePayload(statement="S-FROM-PAYLOAD", correct_answer=True),
        explanation_correct="",
        explanation_incorrect="",
        user_answer=None,
        is_correct=None,
    )
    out = json.loads(format_exercise_history([rec]))
    assert out[0]["statement"] == "S-FROM-PAYLOAD"
    assert out[0]["exercise_type"] == "true_false"
