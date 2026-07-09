import pytest

from api.domains.learning.exercise_types.base import ExerciseTypeHandler
from api.domains.learning.exercise_types.models import ExerciseType
from api.domains.learning.exercise_types.registry import (
    _HANDLER_CLASSES,
    get_handler,
    register,
)


def test_register_and_get_handler_returns_fresh_instances():
    # Registering a dummy under MCQ clobbers the real handler in the global
    # registry; save and restore it so this test does not pollute others that
    # call get_handler(MCQ) / get_prompt_spec(MCQ) later in the same process.
    original = _HANDLER_CLASSES.get(ExerciseType.MCQ)
    try:

        @register
        class _DummyHandler(ExerciseTypeHandler):
            exercise_type = ExerciseType.MCQ
            output_model = object
            payload_model = object

            def prompt_instruction(self):
                return "i"

            def negative_constraints(self):
                return "n"

            def explanation_guidance(self):
                return "e"

            def payload_from_output(self, result):
                return result

            def to_response_dict(self, exercise):
                return {}

            def evaluate(self, exercise, answer):
                return (True, "")

            def tutor_question(self, exercise):
                return ""

            def tutor_options(self, exercise):
                return []

            def serialize_answer(self, exercise, answer):
                return None

        a = get_handler(ExerciseType.MCQ)
        b = get_handler(ExerciseType.MCQ)
        assert isinstance(a, _DummyHandler)
        assert a is not b  # fresh per call
    finally:
        if original is not None:
            _HANDLER_CLASSES[ExerciseType.MCQ] = original


def test_get_handler_unknown_type_raises_keyerror():
    class _Fake:
        value = "does-not-exist"

    with pytest.raises(KeyError):
        get_handler(_Fake())
