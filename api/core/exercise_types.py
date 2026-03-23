"""
exercise_types.py — Shared exercise type schemas, selection, and serialization.
"""

from typing import Any, Dict, Literal, Optional, Sequence

from pydantic import BaseModel, Field


BLOOM_VERBS = {
    1: "Remember (Nhớ: Định nghĩa, liệt kê, ghi nhớ)",
    2: "Understand (Hiểu: Giải thích, tóm tắt)",
    3: "Apply (Vận dụng: Tính toán, áp dụng công thức)",
    4: "Analyze (Phân tích: So sánh, đối chiếu, chia nhỏ vấn đề)",
    5: "Evaluate (Đánh giá: Biện luận, phán xét tính đúng sai)",
    6: "Create (Sáng tạo: Thiết kế, chứng minh, tổng hợp)",
}

ExerciseType = Literal[
    "mcq",
    "true_false",
    "fill_blank",
    "multi_correct",
    "ordering",
    "matching",
    "short_answer",
]


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
    right_items: list[str] = Field(..., min_length=3, description="Right-column choices in shuffled order")


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
            "question": result.sentence,
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
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "items": result.items,
            "correct_answer": result.correct_order,
            "correct_option": join_lines(result.correct_order),
            "explanation_correct": result.explanation_correct,
            "explanation_incorrect": result.explanation_incorrect,
        }

    if isinstance(result, MatchingOutput):
        pairs = [{"left": item.left, "right": item.right} for item in result.pairs]
        return {
            "exercise_type": result.exercise_type,
            "question": result.question,
            "pairs": pairs,
            "left_items": [item["left"] for item in pairs],
            "right_items": result.right_items,
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


def select_exercise_type(bloom_level: int, mastery: Optional[float] = None) -> ExerciseType:
    mastery_value = 0.5 if mastery is None else max(0.0, min(1.0, float(mastery)))

    if bloom_level <= 1:
        return "true_false" if mastery_value < 0.55 else "mcq"
    if bloom_level == 2:
        if mastery_value < 0.3:
            return "true_false"
        if mastery_value < 0.55:
            return "fill_blank"
        if mastery_value < 0.8:
            return "matching"
        return "mcq"
    if bloom_level == 3:
        if mastery_value < 0.35:
            return "fill_blank"
        if mastery_value < 0.7:
            return "matching"
        return "ordering"
    if bloom_level == 4:
        return "ordering" if mastery_value < 0.5 else "multi_correct"
    if bloom_level == 5:
        return "multi_correct" if mastery_value < 0.7 else "short_answer"
    return "mcq" if mastery_value < 0.35 else "short_answer"
