"""
models.py — Shared exercise type enums and LM-output schemas.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

BLOOM_VERBS = {
    1: "Remember (Nhớ: Định nghĩa, liệt kê, ghi nhớ)",
    2: "Understand (Hiểu: Giải thích, tóm tắt)",
    3: "Apply (Vận dụng: Tính toán, áp dụng công thức)",
    4: "Analyze (Phân tích: So sánh, đối chiếu, chia nhỏ vấn đề)",
    5: "Evaluate (Đánh giá: Biện luận, phán xét tính đúng sai)",
    6: "Create (Sáng tạo: Thiết kế, chứng minh, tổng hợp)",
}


class ExerciseType(StrEnum):
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
    explanation_incorrect: str = Field(
        ..., description="Explanation shown when learner is incorrect"
    )


class ExerciseOptions(BaseModel):
    A: str = Field(..., description="Option A")
    B: str = Field(..., description="Option B")
    C: str = Field(..., description="Option C")
    D: str = Field(..., description="Option D")


class MCQOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.MCQ] = ExerciseType.MCQ
    options: ExerciseOptions = Field(..., description="Four options A/B/C/D")
    correct_option: Literal["A", "B", "C", "D"] = Field(..., description="Correct option label")


class TrueFalseOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.TRUE_FALSE] = ExerciseType.TRUE_FALSE
    statement: str = Field(
        ..., description="A single statement the learner judges as true or false"
    )
    correct_answer: bool = Field(..., description="Whether the statement is true")


class FillBlankOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.FILL_BLANK] = ExerciseType.FILL_BLANK
    sentence: str = Field(
        ..., description="Sentence containing exactly one blank placeholder _____"
    )
    blank_answers: list[str] = Field(
        ..., min_length=1, description="Accepted answers for the blank"
    )
    hint: str = Field(..., description="Short hint to guide the learner")


class ExerciseOptionsFive(BaseModel):
    A: str = Field(..., description="Option A")
    B: str = Field(..., description="Option B")
    C: str = Field(..., description="Option C")
    D: str = Field(..., description="Option D")
    E: str = Field(..., description="Option E")


class MultiCorrectOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.MULTI_CORRECT] = ExerciseType.MULTI_CORRECT
    options: ExerciseOptionsFive = Field(..., description="Five options A-E")
    correct_options: list[Literal["A", "B", "C", "D", "E"]] = Field(
        ...,
        min_length=2,
        description="All correct option labels",
    )


class OrderingOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.ORDERING] = ExerciseType.ORDERING
    items: list[str] = Field(
        ..., min_length=3, description="Items shown to the learner in scrambled order"
    )
    correct_order: list[str] = Field(
        ..., min_length=3, description="Same items arranged in the correct order"
    )


class MatchingPair(BaseModel):
    left: str = Field(..., description="Prompt item shown in the left column")
    right: str = Field(..., description="Correct match shown in the right column")


class MatchingOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.MATCHING] = ExerciseType.MATCHING
    pairs: list[MatchingPair] = Field(..., min_length=3, description="Correct left-right pairs")


class ShortAnswerOutput(ExerciseBaseOutput):
    exercise_type: Literal[ExerciseType.SHORT_ANSWER] = ExerciseType.SHORT_ANSWER
    rubric: list[str] = Field(..., min_length=2, description="Short grading rubric bullets")
    sample_answer: str = Field(..., description="Reference answer for grading and review")


class ShortAnswerEvaluationOutput(BaseModel):
    is_correct: bool = Field(..., description="Whether the student's answer satisfies the rubric")
    explanation: str = Field(..., description="Feedback in Vietnamese, grounded in the rubric")
    score: int = Field(..., ge=0, le=10, description="Holistic score from 0 to 10")
