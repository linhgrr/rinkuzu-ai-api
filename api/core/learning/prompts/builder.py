from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from api.core.learning.exercise_types import BLOOM_VERBS, ExerciseType
from api.core.learning.history_formatter import format_exercise_history

from .constants import (
    BLOOM_LEVEL_GUIDANCE,
    COMMON_RULES,
    EMPTY_DEFINITION_GUARD,
    EXERCISE_TYPE_BLOOM_GUIDANCE,
    EXPLANATION_TONE_GUIDANCE,
    MATH_FORMAT_RULES,
    META_VALIDATION_CHECKLIST,
)
from .few_shots import FEW_SHOT_EXAMPLES, FEW_SHOT_HIGH_BLOOM, FEW_SHOT_NON_STEM
from .registry import get_prompt_spec

if TYPE_CHECKING:
    from collections.abc import Sequence


class PromptBuilder:
    """Compose exercise generation prompts as structured message lists."""

    def __init__(
        self,
        concept_name: str,
        concept_definition: str,
        bloom_level: int,
        exercise_type: ExerciseType,
        subject_context: str = "",
    ) -> None:
        self.concept_name = concept_name
        self.concept_definition = concept_definition or ""
        self.bloom_level = bloom_level
        self.exercise_type = exercise_type
        self.subject_context = subject_context
        self.spec = get_prompt_spec(exercise_type)

    def build_system_message(self) -> str:
        bloom_guidance = BLOOM_LEVEL_GUIDANCE.get(self.bloom_level, "")
        type_bloom_guidance = EXERCISE_TYPE_BLOOM_GUIDANCE.get(self.exercise_type, {}).get(
            self.bloom_level,
            "",
        )
        return "\n\n".join(
            part
            for part in [
                "Bạn là giáo viên chuyên tạo bài tập adaptive learning theo Bloom's Taxonomy.",
                COMMON_RULES,
                MATH_FORMAT_RULES,
                bloom_guidance,
                type_bloom_guidance,
                EXPLANATION_TONE_GUIDANCE,
            ]
            if part
        )

    def _select_few_shots(self) -> list[dict[str, Any]]:
        """Select appropriate few-shot examples based on bloom level and subject."""
        shots: list[dict[str, Any]] = []

        # Always include the primary example
        primary = FEW_SHOT_EXAMPLES.get(self.exercise_type)
        if primary:
            shots.append(primary)

        # Append high-bloom example when bloom >= 4
        if self.bloom_level >= 4:
            high_bloom = FEW_SHOT_HIGH_BLOOM.get(self.exercise_type)
            if high_bloom:
                shots.append(high_bloom)

        # Append non-STEM example when subject context suggests it
        non_stem_keywords = (
            "văn", "sử", "địa", "gdcd", "giáo dục", "tiếng",
            "lịch sử", "ngữ văn", "triết", "chính trị", "xã hội",
        )
        context_lower = (self.subject_context + " " + self.concept_name).lower()
        if any(kw in context_lower for kw in non_stem_keywords):
            non_stem = FEW_SHOT_NON_STEM.get(self.exercise_type)
            if non_stem:
                shots.append(non_stem)

        return shots

    def build_user_message(
        self,
        type_spec: str,
        negative_constraints: str,
        recent_exercises: Sequence[dict[str, Any]] | None,
    ) -> str:
        sections: list[str] = []

        # --- Concept section ---
        concept_lines = [
            f"Kiến thức: {self.concept_name}",
            f"Định nghĩa kiến thức: {self.concept_definition or '(không có)'}",
        ]
        if self.subject_context:
            concept_lines.append(f"Ngữ cảnh môn học: {self.subject_context}")
        concept_lines.append(
            f"Bloom level: {self.bloom_level} - {BLOOM_VERBS.get(self.bloom_level, '')}"
        )
        sections.append(
            "<concept>\n" + "\n".join(concept_lines) + "\n</concept>"
        )

        # --- Task instruction ---
        sections.append(
            "<task>\n" + type_spec + "\n</task>"
        )

        # --- Constraints ---
        sections.append(
            "<constraints>\n"
             "Ràng buộc negative constraints:\n" + negative_constraints + "\n"
            + "Hướng dẫn explanation:\n" + self.spec.explanation_guidance + "\n"
            + "</constraints>"
        )

        # --- Empty definition guard ---
        if len(self.concept_definition.strip()) < 20:
            sections.append(
                "<warning>\n" + EMPTY_DEFINITION_GUARD + "</warning>"
            )

        # --- Few-shot examples ---
        few_shots = self._select_few_shots()
        if few_shots:
            examples_json = "\n---\n".join(
                json.dumps(shot, ensure_ascii=False, indent=2) for shot in few_shots
            )
            sections.append(
                "<few_shot>\nFew-shot JSON mẫu:\n" + examples_json + "\n</few_shot>"
            )

        # --- Recent exercise history & diversity enforcement ---
        if recent_exercises:
            diversity_rules = (
                "YÊU CẦU ĐA DẠNG HÓA BẮT BUỘC:\n"
                "- KHÔNG lặp lại cùng góc hỏi, cùng cách diễn đạt, hoặc cùng cấu trúc câu hỏi.\n"
                "- Thay đổi ngữ cảnh/tình huống: nếu bài trước dùng ví dụ A, hãy dùng ví dụ B hoàn toàn khác.\n"
                "- Thay đổi chiến lược distractor: nếu bài trước dùng lỗi tính toán, hãy dùng lỗi khái niệm.\n"
                "- Thay đổi dạng câu: nếu bài trước hỏi 'cái nào đúng', hãy hỏi 'cái nào SAI' hoặc hỏi ngược.\n"
                "- Thay đổi độ trừu tượng: nếu bài trước dùng số cụ thể, hãy dùng biến hoặc ngược lại.\n"
                "- Ưu tiên khai thác khía cạnh chưa được hỏi của kiến thức.\n"
            )
            sections.append(
                "<history>\n"
                + diversity_rules + "\n"
                + "Các bài gần nhất cùng concept (TRÁNH trùng lặp với các bài dưới đây):\n"
                + format_exercise_history(recent_exercises)
                + "\n</history>"
            )

        # --- Meta checklist ---
        sections.append(
            "<checklist>\n" + META_VALIDATION_CHECKLIST + "</checklist>"
        )

        return "\n\n".join(sections)

    def build_messages(
        self,
        recent_exercises: Sequence[dict[str, Any]] | None = None,
    ) -> list[BaseMessage]:
        return [
            SystemMessage(content=self.build_system_message()),
            HumanMessage(
                content=self.build_user_message(
                    type_spec=self.spec.instruction,
                    negative_constraints=self.spec.negative_constraints,
                    recent_exercises=recent_exercises,
                )
            ),
        ]


def build_exercise_messages(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    exercise_type: ExerciseType,
    recent_exercises: Sequence[dict[str, Any]] | None = None,
    subject_context: str = "",
) -> list[BaseMessage]:
    builder = PromptBuilder(
        concept_name=concept_name,
        concept_definition=concept_definition,
        bloom_level=bloom_level,
        exercise_type=exercise_type,
        subject_context=subject_context,
    )
    return builder.build_messages(recent_exercises=recent_exercises)
