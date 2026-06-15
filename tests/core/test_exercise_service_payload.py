# tests/core/test_exercise_service_payload.py

from pydantic import TypeAdapter

from api.core.learning.exercise_types.payloads import ExercisePayload

_adapter = TypeAdapter(ExercisePayload)


def test_payload_validates_from_generation_dict():
    # The exact dict shape generate_exercise now emits under "payload".
    data = {
        "exercise_type": "mcq",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "correct_option": "A",
    }
    payload = _adapter.validate_python(data)
    assert payload.exercise_type == "mcq"
    assert payload.correct_option == "A"
