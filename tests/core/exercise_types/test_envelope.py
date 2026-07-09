from api.domains.learning.exercise_types.payloads import MCQPayload
from api.domains.learning.session import ExerciseRecord


def test_exercise_record_holds_payload():
    rec = ExerciseRecord(
        exercise_id="ex1",
        concept_idx=0,
        concept_name="C",
        bloom_level=1,
        question="Q",
        payload=MCQPayload(options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option="A"),
    )
    assert rec.payload.exercise_type == "mcq"
    assert rec.payload.correct_option == "A"
