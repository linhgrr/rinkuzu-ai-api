"""
selection.py — Exercise type selection and shared text helpers.
"""

from collections.abc import Sequence
import secrets

from .models import ExerciseType

# Use a SystemRandom instance for non-cryptographic educational selection.
_rng = secrets.SystemRandom()

# Mastery bin thresholds for exercise type selection.
_LOW_MASTERY_THRESHOLD = 0.4
_HIGH_MASTERY_THRESHOLD = 0.7


def join_lines(values: Sequence[str]) -> str:
    return "\n".join(f"{index + 1}. {value}" for index, value in enumerate(values))


def normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


EXERCISE_WEIGHTS: dict[int, dict[ExerciseType, tuple[int, int, int]]] = {
    1: {
        ExerciseType.TRUE_FALSE: (70, 40, 10),
        ExerciseType.MCQ: (30, 60, 90),
    },
    2: {
        ExerciseType.TRUE_FALSE: (60, 20, 5),
        ExerciseType.MCQ: (30, 40, 25),
        ExerciseType.FILL_BLANK: (10, 30, 40),
        ExerciseType.MATCHING: (0, 10, 30),
    },
    3: {
        ExerciseType.MCQ: (50, 20, 5),
        ExerciseType.FILL_BLANK: (30, 40, 20),
        ExerciseType.MATCHING: (20, 30, 20),
        ExerciseType.MULTI_CORRECT: (0, 10, 35),
        ExerciseType.ORDERING: (0, 0, 20),
    },
    4: {
        ExerciseType.ORDERING: (70, 40, 20),
        ExerciseType.MULTI_CORRECT: (30, 60, 80),
    },
    5: {
        ExerciseType.MULTI_CORRECT: (80, 50, 20),
        ExerciseType.SHORT_ANSWER: (20, 50, 80),
    },
    6: {
        ExerciseType.MCQ: (60, 30, 0),
        ExerciseType.SHORT_ANSWER: (40, 70, 100),
    },
}


def select_exercise_type(bloom_level: int, mastery: float | None = None) -> ExerciseType:
    mastery_value = 0.5 if mastery is None else max(0.0, min(1.0, float(mastery)))
    bloom_level = max(1, min(6, bloom_level))

    weights_config = EXERCISE_WEIGHTS.get(bloom_level, EXERCISE_WEIGHTS[1])

    # Determine the mastery bin index: 0 (Low), 1 (Mid), 2 (High)
    if mastery_value < _LOW_MASTERY_THRESHOLD:
        weight_index = 0
    elif mastery_value < _HIGH_MASTERY_THRESHOLD:
        weight_index = 1
    else:
        weight_index = 2

    candidates: list[ExerciseType] = []
    weights: list[int] = []

    for ex_type, w_tuple in weights_config.items():
        w = w_tuple[weight_index]
        if w > 0:
            candidates.append(ex_type)
            weights.append(w)

    if not candidates:
        return ExerciseType.MCQ  # Safe fallback

    # random.choices returns a list of k elements, we pluck the first
    return _rng.choices(candidates, weights=weights, k=1)[0]
