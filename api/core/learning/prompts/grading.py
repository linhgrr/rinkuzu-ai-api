from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .constants import (
    MATH_FORMAT_RULES,
    SCORE_ANCHORS,
    THEORY_EXAMPLES_CONSTRAINT,
)


class TheoryOutput(BaseModel):
    content: str = Field(..., description="Concise theory summary in Vietnamese")
    examples: list[str] = Field(..., description="2-3 illustrative examples in Vietnamese")


def build_grading_messages(
    question: str,
    rubric: Sequence[str],
    sample_answer: str,
    student_answer: str,
) -> list[BaseMessage]:
    rubric_lines = "\n".join(f"- {item}" for item in rubric)
    return [
        SystemMessage(
            content=(
                "Bạn là giáo viên chấm câu trả lời ngắn theo rubric.\n"
                "- Chấm khách quan theo rubric được cung cấp.\n"
                "- Không phạt vì khác wording nếu học sinh đúng bản chất.\n"
                "- Feedback phải chỉ rõ ý nào đạt, ý nào thiếu.\n"
                "- Feedback dùng giọng nhẹ nhàng, hướng dẫn. KHÔNG phê phán.\n\n"
                f"{SCORE_ANCHORS}\n\n"
                f"{MATH_FORMAT_RULES}"
            )
        ),
        HumanMessage(
            content=(
                f"<question>\n{question}\n</question>\n\n"
                f"<rubric>\n{rubric_lines}\n</rubric>\n\n"
                f"<sample_answer>\n{sample_answer}\n</sample_answer>\n\n"
                f"<student_answer>\n{student_answer}\n</student_answer>\n\n"
                "Hãy trả về đánh giá theo schema."
            )
        ),
    ]


def build_theory_messages(
    concept_name: str,
    concept_definition: str,
    bloom_level: int = 2,
) -> list[BaseMessage]:
    system_parts = [
        "Bạn là chuyên gia giáo dục chuyên viết phần lý thuyết ngắn gọn, rõ ràng.\n"
        "- Nội dung phải dễ hiểu, đúng bản chất.\n"
        "- Ví dụ phải cụ thể và dùng được ngay.",
        THEORY_EXAMPLES_CONSTRAINT,
        MATH_FORMAT_RULES,
    ]

    return [
        SystemMessage(content="\n\n".join(system_parts)),
        HumanMessage(
            content=(
                f"<concept>\n"
                f"Tên chủ đề: {concept_name}\n"
                f"Định nghĩa / nội dung chính: {concept_definition}\n"
                f"Mức Bloom tham chiếu: {bloom_level}\n"
                f"</concept>\n\n"
                "Hãy tạo `content` tóm tắt lý thuyết và `examples` gồm các ví dụ cụ thể có lời giải ngắn."
            )
        ),
    ]
