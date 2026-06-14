"""
selection.py — Exercise type selection and result serialization helpers.
"""

from collections.abc import Sequence
import secrets
from typing import cast

from .models import (
    ExerciseBaseOutput,
    ExerciseType,
    FillBlankOutput,
    MatchingOutput,
    MCQOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerOutput,
    TrueFalseOutput,
)

# Use a SystemRandom instance for non-cryptographic educational shuffling.
_rng = secrets.SystemRandom()

# Mastery bin thresholds for exercise type selection.
_LOW_MASTERY_THRESHOLD = 0.4
_HIGH_MASTERY_THRESHOLD = 0.7


def join_lines(values: Sequence[str]) -> str:
    return "\n".join(f"{index + 1}. {value}" for index, value in enumerate(values))


def shuffle_ordering_items(correct_order: Sequence[str], max_attempts: int = 5) -> list[str]:
    items = list(correct_order)
    if len(items) <= 1:
        return items

    shuffled = items[:]
    for _ in range(max_attempts):
        _rng.shuffle(shuffled)
        if shuffled != items:
            return shuffled

    # Guard: if repeated shuffles still match the correct order, force a simple rotation.
    return items[1:] + items[:1]


def serialize_exercise_result(
    result: ExerciseBaseOutput,
) -> dict[str, str | bool | list[str] | dict[str, str] | list[dict[str, str]]]:
    if isinstance(result, MCQOutput):
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "options": {
                "A": result.options.A,
                "B": result.options.B,
                "C": result.options.C,
                "D": result.options.D,
            },
            "correct_option": result.correct_option,
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, TrueFalseOutput):
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "statement": result.statement,
            "correct_answer": result.correct_answer,
            "correct_option": "True" if result.correct_answer else "False",
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, FillBlankOutput):
        accepted_answers = [answer.strip() for answer in result.blank_answers if answer.strip()]
        canonical_answer = accepted_answers[0] if accepted_answers else ""
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "sentence": result.sentence,
            "hint": result.hint,
            "blank_answers": accepted_answers,
            "correct_answer": accepted_answers,
            "correct_option": canonical_answer,
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, MultiCorrectOutput):
        correct_options = cast("list[str]", sorted(set(result.correct_options)))
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "options": {
                "A": result.options.A,
                "B": result.options.B,
                "C": result.options.C,
                "D": result.options.D,
                "E": result.options.E,
            },
            "correct_answer": correct_options,
            "correct_option": ", ".join(correct_options),
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, OrderingOutput):
        correct_order = [item.strip() for item in result.correct_order if item.strip()]
        items = shuffle_ordering_items(correct_order)
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "items": items,
            "correct_answer": correct_order,
            "correct_option": join_lines(correct_order),
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, MatchingOutput):
        pairs = [{"left": item.left, "right": item.right} for item in result.pairs]
        right_items = [item["right"] for item in pairs]
        _rng.shuffle(right_items)
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "pairs": pairs,
            "left_items": [item["left"] for item in pairs],
            "right_items": right_items,
            "correct_answer": {item["left"]: item["right"] for item in pairs},
            "correct_option": join_lines([f"{item['left']} → {item['right']}" for item in pairs]),
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, ShortAnswerOutput):
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "rubric": result.rubric,
            "sample_answer": result.sample_answer,
            "correct_answer": result.sample_answer,
            "correct_option": result.sample_answer,
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    raise TypeError(f"Unsupported exercise output type: {type(result)}")


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
