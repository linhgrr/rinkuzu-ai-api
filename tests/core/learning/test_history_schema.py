from pydantic import ValidationError
import pytest

from api.domains.learning.schemas import ExerciseHistoryResponse


def test_exercise_history_rejects_empty_exercise_id() -> None:
    with pytest.raises(ValidationError, match="exercise_id"):
        ExerciseHistoryResponse.model_validate(
            {
                "exercise_id": "",
                "concept_idx": 0,
                "concept_name": "Concept",
                "bloom_level": 1,
                "question": "Question?",
                "explanation": "Explanation",
                "payload": {
                    "exercise_type": "mcq",
                    "options": {"A": "Answer"},
                    "correct_option": "A",
                },
                "explanation_correct": "Correct",
                "explanation_incorrect": "Incorrect",
                "theory": None,
                "user_answer": "A",
                "is_correct": True,
                "timestamp": 100.0,
            }
        )
