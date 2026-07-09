from api.domains.learning.exercise_types import (
    ExerciseType,
    FillBlankOutput,
    MatchingOutput,
    MCQOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerOutput,
    TrueFalseOutput,
)
from api.domains.learning.prompts.registry import get_prompt_spec


def test_all_exercise_types_have_spec():
    for exercise_type in ExerciseType:
        assert get_prompt_spec(exercise_type) is not None


def test_get_prompt_spec_returns_correct_schema():
    assert get_prompt_spec(ExerciseType.MCQ).schema is MCQOutput
    assert get_prompt_spec(ExerciseType.TRUE_FALSE).schema is TrueFalseOutput
    assert get_prompt_spec(ExerciseType.FILL_BLANK).schema is FillBlankOutput
    assert get_prompt_spec(ExerciseType.MULTI_CORRECT).schema is MultiCorrectOutput
    assert get_prompt_spec(ExerciseType.ORDERING).schema is OrderingOutput
    assert get_prompt_spec(ExerciseType.MATCHING).schema is MatchingOutput
    assert get_prompt_spec(ExerciseType.SHORT_ANSWER).schema is ShortAnswerOutput


def test_spec_has_negative_constraints():
    for exercise_type in ExerciseType:
        spec = get_prompt_spec(exercise_type)
        assert spec.negative_constraints.strip()
