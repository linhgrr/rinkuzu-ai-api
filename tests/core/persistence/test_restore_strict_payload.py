from pydantic import ValidationError
import pytest

from api.domains.learning.session import SessionState


def _doc_entry(payload: dict) -> dict:
    return {
        "exercise_id": "ex1",
        "concept_idx": 0,
        "concept_name": "C",
        "bloom_level": 1,
        "question": "Q",
        "explanation": "",
        "explanation_correct": "",
        "explanation_incorrect": "",
        "payload": payload,
        "theory": None,
        "user_answer": None,
        "is_correct": None,
        "timestamp": 1.0,
    }


def test_restore_builds_typed_payload():
    session = SessionState.__new__(SessionState)
    session.exercise_history = []
    SessionState._restore_exercise_records(
        session,
        [_doc_entry({"exercise_type": "true_false", "statement": "S", "correct_answer": True})],
    )
    rec = session.exercise_history[0]
    assert rec.payload.exercise_type.value == "true_false"
    assert rec.payload.statement == "S"


def test_restore_rejects_malformed_payload():
    session = SessionState.__new__(SessionState)
    session.exercise_history = []
    with pytest.raises(ValidationError):
        SessionState._restore_exercise_records(
            session,
            [_doc_entry({"exercise_type": "true_false"})],  # missing statement
        )
