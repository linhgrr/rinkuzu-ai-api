"""
exercise_types.py — Shared exercise type schemas, selection, and serialization.
"""

from typing import Any, Dict, Literal, Optional, Sequence, cast
import random

from pydantic import BaseModel, Field


BLOOM_VERBS = {
    1: "Remember (Nhớ: Định nghĩa, liệt kê, ghi nhớ)",
    2: "Understand (Hiểu: Giải thích, tóm tắt)",
    3: "Apply (Vận dụng: Tính toán, áp dụng công thức)",
    4: "Analyze (Phân tích: So sánh, đối chiếu, chia nhỏ vấn đề)",
    5: "Evaluate (Đánh giá: Biện luận, phán xét tính đúng sai)",
    6: "Create (Sáng tạo: Thiết kế, chứng minh, tổng hợp)",
}

from enum import Enum

class ExerciseType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    MULTI_CORRECT = "multi_correct"
    ORDERING = "ordering"
    MATCHING = "matching"
    SHORT_ANSWER = "short_answer"


class ExerciseBaseOutput(BaseModel):
    exercise_type: ExerciseType
    question: str = Field(..., description="Question or instruction text")
    explanation_correct: str = Field(..., description="Explanation shown when learner is correct")
    explanation_incorrect: str = Field(..., description="Explanation shown when learner is incorrect")


class ExerciseOptions(BaseModel):
    A: str = Field(..., description="Option A")
    B: str = Field(..., description="Option B")
    C: str = Field(..., description="Option C")
    D: str = Field(..., description="Option D")


class MCQOutput(ExerciseBaseOutput):
    exercise_type: Literal["mcq"] = "mcq"
    options: ExerciseOptions = Field(..., description="Four options A/B/C/D")
    correct_option: Literal["A", "B", "C", "D"] = Field(..., description="Correct option label")


class TrueFalseOutput(ExerciseBaseOutput):
    exercise_type: Literal["true_false"] = "true_false"
    statement: str = Field(..., description="A single statement the learner judges as true or false")
    correct_answer: bool = Field(..., description="Whether the statement is true")


class FillBlankOutput(ExerciseBaseOutput):
    exercise_type: Literal["fill_blank"] = "fill_blank"
    sentence: str = Field(..., description="Sentence containing exactly one blank placeholder _____")
    blank_answers: list[str] = Field(..., min_length=1, description="Accepted answers for the blank")
    hint: str = Field(..., description="Short hint to guide the learner")


class ExerciseOptionsFive(BaseModel):
    A: str = Field(..., description="Option A")
    B: str = Field(..., description="Option B")
    C: str = Field(..., description="Option C")
    D: str = Field(..., description="Option D")
    E: str = Field(..., description="Option E")


class MultiCorrectOutput(ExerciseBaseOutput):
    exercise_type: Literal["multi_correct"] = "multi_correct"
    options: ExerciseOptionsFive = Field(..., description="Five options A-E")
    correct_options: list[Literal["A", "B", "C", "D", "E"]] = Field(
        ...,
        min_length=2,
        description="All correct option labels",
    )


class OrderingOutput(ExerciseBaseOutput):
    exercise_type: Literal["ordering"] = "ordering"
    items: list[str] = Field(..., min_length=3, description="Items shown to the learner in scrambled order")
    correct_order: list[str] = Field(..., min_length=3, description="Same items arranged in the correct order")


class MatchingPair(BaseModel):
    left: str = Field(..., description="Prompt item shown in the left column")
    right: str = Field(..., description="Correct match shown in the right column")


class MatchingOutput(ExerciseBaseOutput):
    exercise_type: Literal["matching"] = "matching"
    pairs: list[MatchingPair] = Field(..., min_length=3, description="Correct left-right pairs")


class ShortAnswerOutput(ExerciseBaseOutput):
    exercise_type: Literal["short_answer"] = "short_answer"
    rubric: list[str] = Field(..., min_length=2, description="Short grading rubric bullets")
    sample_answer: str = Field(..., description="Reference answer for grading and review")


class ShortAnswerEvaluationOutput(BaseModel):
    is_correct: bool = Field(..., description="Whether the student's answer satisfies the rubric")
    explanation: str = Field(..., description="Feedback in Vietnamese, grounded in the rubric")
    score: int = Field(..., ge=0, le=10, description="Holistic score from 0 to 10")


def join_lines(values: Sequence[str]) -> str:
    return "\n".join(f"{index + 1}. {value}" for index, value in enumerate(values))


def shuffle_ordering_items(correct_order: Sequence[str], max_attempts: int = 5) -> list[str]:
    items = list(correct_order)
    if len(items) <= 1:
        return items

    shuffled = items[:]
    for _ in range(max_attempts):
        random.shuffle(shuffled)
        if shuffled != items:
            return shuffled

    # Guard: if repeated shuffles still match the correct order, force a simple rotation.
    return items[1:] + items[:1]


def serialize_exercise_result(result: ExerciseBaseOutput) -> Dict[str, Any]:
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
        correct_options = sorted(set(result.correct_options))
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
        random.shuffle(right_items)
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


# Configuration-driven weight matrix for exercise type selection.
# Structure: { bloom_level: { exercise_type: (weight_low_mastery, weight_mid_mastery, weight_high_mastery) } }
# Mastery bins: Low (< 0.4), Mid (0.4 - 0.7), High (>= 0.7)
EXERCISE_WEIGHTS: Dict[int, Dict[ExerciseType, tuple[int, int, int]]] = {
    1: {
        ExerciseType.TRUE_FALSE: (70, 40, 10),
        ExerciseType.MCQ:        (30, 60, 90),
    },
    2: {
        ExerciseType.TRUE_FALSE: (60, 20,  5),
        ExerciseType.MCQ:        (30, 40, 25),
        ExerciseType.FILL_BLANK: (10, 30, 40),
        ExerciseType.MATCHING:   ( 0, 10, 30),
    },
    3: {
        ExerciseType.MCQ:           (50, 20,  5),
        ExerciseType.FILL_BLANK:    (30, 40, 20),
        ExerciseType.MATCHING:      (20, 30, 20),
        ExerciseType.MULTI_CORRECT: ( 0, 10, 35),
        ExerciseType.ORDERING:      ( 0,  0, 20),
    },
    4: {
        ExerciseType.ORDERING:      (70, 40, 20),
        ExerciseType.MULTI_CORRECT: (30, 60, 80),
    },
    5: {
        ExerciseType.MULTI_CORRECT: (80, 50, 20),
        ExerciseType.SHORT_ANSWER:  (20, 50, 80),
    },
    6: {
        ExerciseType.MCQ:           (60, 30,  0),
        ExerciseType.SHORT_ANSWER:  (40, 70, 100),
    }
}


def select_exercise_type(bloom_level: int, mastery: Optional[float] = None) -> ExerciseType:
    mastery_value = 0.5 if mastery is None else max(0.0, min(1.0, float(mastery)))
    bloom_level = max(1, min(6, bloom_level))
    
    weights_config = EXERCISE_WEIGHTS.get(bloom_level, EXERCISE_WEIGHTS[1])
    
    # Determine the mastery bin index: 0 (Low), 1 (Mid), 2 (High)
    if mastery_value < 0.4:
        weight_index = 0
    elif mastery_value < 0.7:
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
    selected_type = random.choices(candidates, weights=weights, k=1)[0]
    return selected_type
