from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from api.core.learning.exercise_types import (
    ExerciseType,
    FillBlankOutput,
    MatchingOutput,
    MCQOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerOutput,
    TrueFalseOutput,
    serialize_exercise_result,
)

from .constants import EXPLANATION_GUIDANCE, NEGATIVE_CONSTRAINTS

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel


@dataclass(frozen=True)
class ExercisePromptSpec:
    schema: type[BaseModel]
    instruction: str
    negative_constraints: str
    explanation_guidance: str
    serializer: Callable[[BaseModel], dict[str, Any]]


PROMPT_REGISTRY: dict[ExerciseType, ExercisePromptSpec] = {
    ExerciseType.MCQ: ExercisePromptSpec(
        schema=MCQOutput,
        instruction=(
            "Hãy tạo 1 câu hỏi trắc nghiệm khách quan gồm đúng 4 đáp án A, B, C, D.\n"
            "- Có duy nhất 1 đáp án đúng.\n"
            "- Distractor phải hợp lý và đủ gần để học sinh có thể nhầm nếu hiểu chưa chắc.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.MCQ],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.MCQ],
        serializer=serialize_exercise_result,
    ),
    ExerciseType.TRUE_FALSE: ExercisePromptSpec(
        schema=TrueFalseOutput,
        instruction=(
            "Hãy tạo 1 bài tập dạng Đúng/Sai.\n"
            "- `statement` là một mệnh đề duy nhất để học sinh đánh giá.\n"
            "- `question` là lời dẫn ngắn yêu cầu chọn đúng hoặc sai.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.TRUE_FALSE],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.TRUE_FALSE],
        serializer=serialize_exercise_result,
    ),
    ExerciseType.FILL_BLANK: ExercisePromptSpec(
        schema=FillBlankOutput,
        instruction=(
            "Hãy tạo 1 bài tập điền vào chỗ trống.\n"
            "- `sentence` phải chứa đúng 1 chỗ trống ký hiệu là `_____`.\n"
            "- `blank_answers` gồm 1-3 đáp án tương đương được chấp nhận.\n"
            "- `hint` ngắn gọn nhưng không lộ đáp án.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.FILL_BLANK],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.FILL_BLANK],
        serializer=serialize_exercise_result,
    ),
    ExerciseType.MULTI_CORRECT: ExercisePromptSpec(
        schema=MultiCorrectOutput,
        instruction=(
            "Hãy tạo 1 câu hỏi trắc nghiệm nhiều đáp án đúng gồm đúng 5 lựa chọn A, B, C, D, E.\n"
            "- Số đáp án đúng có thể là 2, 3, hoặc 4 — hãy thoải mái chọn số lượng phù hợp nhất với nội dung câu hỏi.\n"
            "- Các lựa chọn sai phải sai vì thiếu điều kiện hoặc sai bản chất, không được vô lý.\n"
            "- Trước khi output, hãy tự kiểm tra TỪNG lựa chọn A-E: tính toán/suy luận cụ thể để xác nhận đúng hay sai.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.MULTI_CORRECT],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.MULTI_CORRECT],
        serializer=serialize_exercise_result,
    ),
    ExerciseType.ORDERING: ExercisePromptSpec(
        schema=OrderingOutput,
        instruction=(
            "Hãy tạo 1 bài tập sắp xếp thứ tự.\n"
            "- `correct_order` là nguồn chân lý, phải đầy đủ và đúng tuyệt đối.\n"
            "- `items` phải chứa đúng các phần tử của `correct_order`, không thêm bớt.\n"
            "- Nội dung phải chấm được bằng một trình tự duy nhất.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.ORDERING],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.ORDERING],
        serializer=serialize_exercise_result,
    ),
    ExerciseType.MATCHING: ExercisePromptSpec(
        schema=MatchingOutput,
        instruction=(
            "Hãy tạo 1 bài tập ghép nối.\n"
            "- `pairs` gồm 3-5 cặp ghép đúng.\n"
            "- Mỗi `left` chỉ khớp tốt với đúng 1 `right`.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.MATCHING],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.MATCHING],
        serializer=serialize_exercise_result,
    ),
    ExerciseType.SHORT_ANSWER: ExercisePromptSpec(
        schema=ShortAnswerOutput,
        instruction=(
            "Hãy tạo 1 câu hỏi trả lời ngắn để chấm bằng rubric.\n"
            "- `question` phải mở vừa đủ để học sinh diễn đạt, nhưng vẫn chấm được khách quan.\n"
            "- `rubric` gồm 2-4 tiêu chí ngắn, rõ ràng.\n"
            "- `sample_answer` súc tích nhưng bám đủ rubric.\n"
        ),
        negative_constraints=NEGATIVE_CONSTRAINTS[ExerciseType.SHORT_ANSWER],
        explanation_guidance=EXPLANATION_GUIDANCE[ExerciseType.SHORT_ANSWER],
        serializer=serialize_exercise_result,
    ),
}


def get_prompt_spec(exercise_type: ExerciseType) -> ExercisePromptSpec:
    return PROMPT_REGISTRY[exercise_type]
