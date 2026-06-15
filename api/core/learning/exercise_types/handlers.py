"""
handlers.py — One ExerciseTypeHandler per exercise type. Each owns prompt config,
serialization, grading, tutor context, and answer-history for its type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from api.core.learning.prompts.constants import EXPLANATION_GUIDANCE, NEGATIVE_CONSTRAINTS

from .base import ExerciseTypeHandler
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
from .payloads import (
    FillBlankPayload,
    MatchingPayload,
    MCQPayload,
    MultiCorrectPayload,
    OrderingPayload,
    ShortAnswerPayload,
    TrueFalsePayload,
)
from .registry import register
from .selection import join_lines
from .shuffle import deterministic_shuffle

if TYPE_CHECKING:
    from api.core.learning.session import ExerciseRecord


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _payload_of(exercise: ExerciseRecord) -> Any:
    # ExerciseRecord gains a typed ``payload`` field in a later task; until then
    # read it dynamically so this module type-checks against the current record.
    return cast("Any", exercise).payload


# ---- MCQ ----------------------------------------------------------------

_MCQ_INSTRUCTION = (
    "Hãy tạo 1 câu hỏi trắc nghiệm khách quan gồm đúng 4 đáp án A, B, C, D.\n"
    "- Có duy nhất 1 đáp án đúng.\n"
    "- Distractor phải hợp lý và đủ gần để học sinh có thể nhầm nếu hiểu chưa chắc.\n"
)


@register
class MCQHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.MCQ
    output_model = MCQOutput
    payload_model = MCQPayload

    def prompt_instruction(self) -> str:
        return _MCQ_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.MCQ]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.MCQ]

    def payload_from_output(self, result: ExerciseBaseOutput) -> MCQPayload:
        out = cast("MCQOutput", result)
        return MCQPayload(
            options={
                "A": out.options.A,
                "B": out.options.B,
                "C": out.options.C,
                "D": out.options.D,
            },
            correct_option=out.correct_option,
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: MCQPayload = _payload_of(exercise)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "options": dict(payload.options),
            "correct_option": payload.correct_option,
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: MCQPayload = _payload_of(exercise)
        selected = (answer.get("choice") or "").strip().upper()
        return selected == payload.correct_option.strip().upper(), selected

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: MCQPayload = _payload_of(exercise)
        return [payload.options[k] for k in sorted(payload.options) if payload.options.get(k)]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        choices = answer.get("choices") or []
        if choices:
            return ", ".join(sorted(choices))
        return answer.get("choice")


# ---- True/False ---------------------------------------------------------

_TRUE_FALSE_INSTRUCTION = (
    "Hãy tạo 1 bài tập dạng Đúng/Sai.\n"
    "- `statement` là một mệnh đề duy nhất để học sinh đánh giá.\n"
    "- `question` là lời dẫn ngắn yêu cầu chọn đúng hoặc sai.\n"
)


@register
class TrueFalseHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.TRUE_FALSE
    output_model = TrueFalseOutput
    payload_model = TrueFalsePayload

    def prompt_instruction(self) -> str:
        return _TRUE_FALSE_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.TRUE_FALSE]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.TRUE_FALSE]

    def payload_from_output(self, result: ExerciseBaseOutput) -> TrueFalsePayload:
        out = cast("TrueFalseOutput", result)
        return TrueFalsePayload(statement=out.statement, correct_answer=out.correct_answer)

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: TrueFalsePayload = _payload_of(exercise)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "statement": payload.statement,
            "correct_answer": payload.correct_answer,
            "correct_option": "True" if payload.correct_answer else "False",
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: TrueFalsePayload = _payload_of(exercise)
        selected = answer.get("boolean")
        return (
            selected is not None and bool(selected) == bool(payload.correct_answer),
            "True" if selected else "False",
        )

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        payload: TrueFalsePayload = _payload_of(exercise)
        return f"{exercise.question}\n\nPhát biểu: {payload.statement}".strip()

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:  # noqa: ARG002 — contract parity
        return ["True", "False"]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        value = answer.get("boolean")
        return None if value is None else ("True" if value else "False")


# ---- Fill blank ---------------------------------------------------------

_FILL_BLANK_INSTRUCTION = (
    "Hãy tạo 1 bài tập điền vào chỗ trống.\n"
    "- `sentence` phải chứa đúng 1 chỗ trống ký hiệu là `_____`.\n"
    "- `blank_answers` gồm 1-3 đáp án tương đương được chấp nhận.\n"
    "- `hint` ngắn gọn nhưng không lộ đáp án.\n"
)


@register
class FillBlankHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.FILL_BLANK
    output_model = FillBlankOutput
    payload_model = FillBlankPayload

    def prompt_instruction(self) -> str:
        return _FILL_BLANK_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.FILL_BLANK]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.FILL_BLANK]

    def payload_from_output(self, result: ExerciseBaseOutput) -> FillBlankPayload:
        out = cast("FillBlankOutput", result)
        accepted = [a.strip() for a in out.blank_answers if a.strip()]
        return FillBlankPayload(
            sentence=out.sentence,
            hint=out.hint,
            blank_answers=accepted,
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: FillBlankPayload = _payload_of(exercise)
        canonical = payload.blank_answers[0] if payload.blank_answers else ""
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "sentence": payload.sentence,
            "hint": payload.hint,
            "blank_answers": list(payload.blank_answers),
            "correct_answer": list(payload.blank_answers),
            "correct_option": canonical,
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: FillBlankPayload = _payload_of(exercise)
        user = [_normalize(b) for b in (answer.get("blanks") or []) if b and b.strip()]
        accepted = [_normalize(a) for a in payload.blank_answers]
        ok = bool(user and accepted and user[0] in accepted)
        return ok, ", ".join(answer.get("blanks") or [])

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        payload: FillBlankPayload = _payload_of(exercise)
        return f"{exercise.question}\n\nCâu cần điền: {payload.sentence}".strip()

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: FillBlankPayload = _payload_of(exercise)
        return [f"Gợi ý: {payload.hint}"] if payload.hint else []

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        blanks = [b.strip() for b in (answer.get("blanks") or []) if b and b.strip()]
        return ", ".join(blanks) or None


# ---- Multi-correct ------------------------------------------------------

_MULTI_CORRECT_INSTRUCTION = (
    "Hãy tạo 1 câu hỏi trắc nghiệm nhiều đáp án đúng gồm đúng 5 lựa chọn A, B, C, D, E.\n"
    "- Số đáp án đúng có thể là 2, 3, hoặc 4 — hãy thoải mái chọn số lượng phù hợp nhất với nội dung câu hỏi.\n"
    "- Các lựa chọn sai phải sai vì thiếu điều kiện hoặc sai bản chất, không được vô lý.\n"
    "- Trước khi output, hãy tự kiểm tra TỪNG lựa chọn A-E: tính toán/suy luận cụ thể để xác nhận đúng hay sai.\n"
)


@register
class MultiCorrectHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.MULTI_CORRECT
    output_model = MultiCorrectOutput
    payload_model = MultiCorrectPayload

    def prompt_instruction(self) -> str:
        return _MULTI_CORRECT_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.MULTI_CORRECT]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.MULTI_CORRECT]

    def payload_from_output(self, result: ExerciseBaseOutput) -> MultiCorrectPayload:
        out = cast("MultiCorrectOutput", result)
        return MultiCorrectPayload(
            options={
                "A": out.options.A,
                "B": out.options.B,
                "C": out.options.C,
                "D": out.options.D,
                "E": out.options.E,
            },
            correct_options=sorted(set(out.correct_options)),
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: MultiCorrectPayload = _payload_of(exercise)
        correct = sorted(set(payload.correct_options))
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "options": dict(payload.options),
            "correct_answer": correct,
            "correct_option": ", ".join(correct),
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: MultiCorrectPayload = _payload_of(exercise)
        selected = sorted(
            {c.strip().upper() for c in (answer.get("choices") or []) if c and c.strip()}
        )
        expected = sorted({c.strip().upper() for c in payload.correct_options})
        return selected == expected, ", ".join(selected)

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: MultiCorrectPayload = _payload_of(exercise)
        return [payload.options[k] for k in sorted(payload.options) if payload.options.get(k)]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        choices = answer.get("choices") or []
        return ", ".join(sorted(choices)) if choices else None


# ---- Ordering -----------------------------------------------------------

_ORDERING_INSTRUCTION = (
    "Hãy tạo 1 bài tập sắp xếp thứ tự.\n"
    "- `correct_order` là nguồn chân lý, phải đầy đủ và đúng tuyệt đối.\n"
    "- `items` phải chứa đúng các phần tử của `correct_order`, không thêm bớt.\n"
    "- Nội dung phải chấm được bằng một trình tự duy nhất.\n"
)


@register
class OrderingHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.ORDERING
    output_model = OrderingOutput
    payload_model = OrderingPayload

    def prompt_instruction(self) -> str:
        return _ORDERING_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.ORDERING]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.ORDERING]

    def payload_from_output(self, result: ExerciseBaseOutput) -> OrderingPayload:
        out = cast("OrderingOutput", result)
        return OrderingPayload(correct_order=[i.strip() for i in out.correct_order if i.strip()])

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: OrderingPayload = _payload_of(exercise)
        display = deterministic_shuffle(payload.correct_order, exercise.exercise_id)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "items": display,
            "correct_answer": list(payload.correct_order),
            "correct_option": join_lines(payload.correct_order),
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: OrderingPayload = _payload_of(exercise)
        selected = [_normalize(i) for i in (answer.get("ordering") or []) if i and i.strip()]
        expected = [_normalize(i) for i in payload.correct_order]
        return bool(selected) and selected == expected, " → ".join(answer.get("ordering") or [])

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: OrderingPayload = _payload_of(exercise)
        return deterministic_shuffle(payload.correct_order, exercise.exercise_id)

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        ordering = [i.strip() for i in (answer.get("ordering") or []) if i and i.strip()]
        return " → ".join(ordering) or None


# ---- Matching -----------------------------------------------------------

_MATCHING_INSTRUCTION = (
    "Hãy tạo 1 bài tập ghép nối.\n"
    "- `pairs` gồm 3-5 cặp ghép đúng.\n"
    "- Mỗi `left` chỉ khớp tốt với đúng 1 `right`.\n"
)


@register
class MatchingHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.MATCHING
    output_model = MatchingOutput
    payload_model = MatchingPayload

    def prompt_instruction(self) -> str:
        return _MATCHING_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.MATCHING]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.MATCHING]

    def payload_from_output(self, result: ExerciseBaseOutput) -> MatchingPayload:
        out = cast("MatchingOutput", result)
        return MatchingPayload(
            pairs=[{"left": p.left, "right": p.right} for p in out.pairs],
        )

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: MatchingPayload = _payload_of(exercise)
        left_items = [p["left"] for p in payload.pairs]
        right_canonical = [p["right"] for p in payload.pairs]
        right_items = deterministic_shuffle(right_canonical, exercise.exercise_id)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "pairs": [dict(p) for p in payload.pairs],
            "left_items": left_items,
            "right_items": right_items,
            "correct_answer": {p["left"]: p["right"] for p in payload.pairs},
            "correct_option": join_lines([f"{p['left']} → {p['right']}" for p in payload.pairs]),
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        payload: MatchingPayload = _payload_of(exercise)
        selected = {
            _normalize(left): _normalize(right)
            for left, right in (answer.get("matching") or {}).items()
            if left and right
        }
        expected = {_normalize(p["left"]): _normalize(p["right"]) for p in payload.pairs}
        summary = ", ".join(
            f"{left} -> {right}" for left, right in (answer.get("matching") or {}).items()
        )
        return bool(selected) and selected == expected, summary

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: MatchingPayload = _payload_of(exercise)
        right_canonical = [p["right"] for p in payload.pairs]
        return deterministic_shuffle(right_canonical, exercise.exercise_id)

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        matching = answer.get("matching") or {}
        if not matching:
            return None
        return ", ".join(f"{left} -> {right}" for left, right in matching.items())


# ---- Short answer -------------------------------------------------------

_SHORT_ANSWER_INSTRUCTION = (
    "Hãy tạo 1 câu hỏi trả lời ngắn để chấm bằng rubric.\n"
    "- `question` phải mở vừa đủ để học sinh diễn đạt, nhưng vẫn chấm được khách quan.\n"
    "- `rubric` gồm 2-4 tiêu chí ngắn, rõ ràng.\n"
    "- `sample_answer` súc tích nhưng bám đủ rubric.\n"
)


@register
class ShortAnswerHandler(ExerciseTypeHandler):
    exercise_type = ExerciseType.SHORT_ANSWER
    output_model = ShortAnswerOutput
    payload_model = ShortAnswerPayload

    def prompt_instruction(self) -> str:
        return _SHORT_ANSWER_INSTRUCTION

    def negative_constraints(self) -> str:
        return NEGATIVE_CONSTRAINTS[ExerciseType.SHORT_ANSWER]

    def explanation_guidance(self) -> str:
        return EXPLANATION_GUIDANCE[ExerciseType.SHORT_ANSWER]

    def payload_from_output(self, result: ExerciseBaseOutput) -> ShortAnswerPayload:
        out = cast("ShortAnswerOutput", result)
        return ShortAnswerPayload(rubric=list(out.rubric), sample_answer=out.sample_answer)

    def to_response_dict(self, exercise: ExerciseRecord) -> dict[str, Any]:
        payload: ShortAnswerPayload = _payload_of(exercise)
        return {
            "exercise_type": self.exercise_type.value,
            "question": exercise.question,
            "rubric": list(payload.rubric),
            "sample_answer": payload.sample_answer,
            "correct_answer": payload.sample_answer,
            "correct_option": payload.sample_answer,
            "explanation_correct": exercise.explanation_correct,
            "explanation_incorrect": exercise.explanation_incorrect,
        }

    def evaluate(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> tuple[bool, str]:
        if self._grader is None:
            raise RuntimeError("short_answer_grader is required for short_answer exercises")
        payload: ShortAnswerPayload = _payload_of(exercise)
        student = (answer.get("text") or "").strip()
        grading = self._grader(
            concept_name=exercise.concept_name,
            question=exercise.question,
            rubric=payload.rubric,
            sample_answer=payload.sample_answer,
            student_answer=student,
        )
        exercise.explanation_correct = str(grading["explanation"])
        exercise.explanation_incorrect = str(grading["explanation"])
        return bool(grading["is_correct"]), student

    def tutor_question(self, exercise: ExerciseRecord) -> str:
        return exercise.question

    def tutor_options(self, exercise: ExerciseRecord) -> list[str]:
        payload: ShortAnswerPayload = _payload_of(exercise)
        return list(payload.rubric) or ["Trả lời ngắn gọn, bám sát câu hỏi."]

    def serialize_answer(self, exercise: ExerciseRecord, answer: dict[str, Any]) -> str | None:  # noqa: ARG002 — contract parity
        return (answer.get("text") or "").strip() or None
