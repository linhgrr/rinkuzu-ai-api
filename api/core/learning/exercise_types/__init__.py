from .models import (
    BLOOM_VERBS,
    ExerciseBaseOutput,
    ExerciseOptions,
    ExerciseOptionsFive,
    ExerciseType,
    FillBlankOutput,
    MatchingOutput,
    MatchingPair,
    MCQOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerEvaluationOutput,
    ShortAnswerOutput,
    TrueFalseOutput,
)
from .selection import (
    EXERCISE_WEIGHTS,
    join_lines,
    normalize_text,
    select_exercise_type,
)
from .selection import _rng as _rng  # noqa: PLC0414  # re-export: tests patch exercise_types._rng

__all__ = [
    "BLOOM_VERBS",
    "EXERCISE_WEIGHTS",
    "ExerciseBaseOutput",
    "ExerciseOptions",
    "ExerciseOptionsFive",
    "ExerciseType",
    "FillBlankOutput",
    "MCQOutput",
    "MatchingOutput",
    "MatchingPair",
    "MultiCorrectOutput",
    "OrderingOutput",
    "ShortAnswerEvaluationOutput",
    "ShortAnswerOutput",
    "TrueFalseOutput",
    "join_lines",
    "normalize_text",
    "select_exercise_type",
]


from . import handlers  # noqa: F401  (import for @register side effects)
from .base import ExerciseTypeHandler
from .registry import get_handler, register

__all__ += ["ExerciseTypeHandler", "get_handler", "register"]
