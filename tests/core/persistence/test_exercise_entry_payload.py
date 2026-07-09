from datetime import UTC, datetime

from api.shared.persistence.documents import ExerciseEntry


def test_exercise_entry_accepts_nested_payload():
    entry = ExerciseEntry(
        exercise_id="ex1",
        concept_idx=0,
        concept_name="C",
        bloom_level=1,
        question="Q",
        explanation="",
        explanation_correct="",
        explanation_incorrect="",
        payload={"exercise_type": "true_false", "statement": "S", "correct_answer": True},
        timestamp=datetime.now(UTC),
    )
    assert entry.payload["exercise_type"] == "true_false"


def test_exercise_entry_has_no_flat_content_fields():
    names = set(ExerciseEntry.model_fields)
    for flat in (
        "statement",
        "sentence",
        "options",
        "items",
        "pairs",
        "right_items",
        "rubric",
        "correct_option",
        "correct_answer",
        "hint",
    ):
        assert flat not in names
