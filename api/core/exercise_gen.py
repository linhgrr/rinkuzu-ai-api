"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.
"""

import time
from typing import Optional, Dict, Any, List, Sequence

from pydantic import BaseModel, Field
from loguru import logger

from .exercise_types import (
    BLOOM_VERBS,
    ExerciseType,
    MatchingOutput,
    MCQOutput,
    FillBlankOutput,
    MultiCorrectOutput,
    OrderingOutput,
    ShortAnswerEvaluationOutput,
    ShortAnswerOutput,
    TrueFalseOutput,
    select_exercise_type,
    serialize_exercise_result,
)
from .llm import (
    get_structured_llm,
    resolve_retry_policy,
    sleep_before_retry,
)


class TheoryOutput(BaseModel):
    """Theory review payload for Bloom 1 & 2."""
    content: str = Field(..., description="Concise theory summary in Vietnamese")
    examples: List[str] = Field(..., description="2-3 illustrative examples in Vietnamese")


MATH_FORMAT_RULES = (
    "Quy tắc định dạng toán BẮT BUỘC:\n"
    "- Mọi biểu thức toán inline PHẢI đặt trong $...$. Ví dụ: $x^2 + 1$.\n"
    "- Công thức toán riêng dòng (display) PHẢI đặt trong $$...$$. Ví dụ:\n"
    "  $$\\Delta = b^2 - 4ac$$\n"
    "- Không dùng dạng text: vec(...), frac(...), sqrt(...). Bắt buộc dùng LaTeX: $\\vec{...}$, $\\frac{...}{...}$, $\\sqrt{...}$.\n"
    "- Chỉ số với chữ cái Hy Lạp phải viết đúng LaTeX, ví dụ: $n_{\\alpha}$, $\\theta_{\\max}$.\n"
    "- KHÔNG viết công thức toán dạng text thuần. Ví dụ SAI: 'x^2 + 1 = 0'. Ví dụ ĐÚNG: '$x^2 + 1 = 0$'.\n"
    "- Ví dụ đúng hoàn chỉnh: $\\vec{n}_{\\alpha} \\cdot \\vec{n}_{\\eta} = 0$.\n"
)


def _build_generation_spec(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    exercise_type: ExerciseType,
):
    common_intro = (
        "Bạn là giáo viên chuyên tạo bài tập adaptive learning theo Bloom's Taxonomy.\n"
        f"Kiến thức yêu cầu: {concept_name}\n"
        f"Định nghĩa kiến thức: {concept_definition}\n"
        f"Mức độ tư duy (Bloom): Level {bloom_level} - {BLOOM_VERBS.get(bloom_level, '')}\n"
        "Ngôn ngữ: Tiếng Việt.\n"
        "Nội dung phải rõ ràng, không lan man, đúng trọng tâm kiến thức.\n"
        "- explanation_correct: 1-3 câu, giọng thân thiện, giải thích vì sao đáp án đúng.\n"
        "- explanation_incorrect: 1-2 câu, nêu lỗi sai phổ biến hoặc điều kiện cần còn thiếu.\n"
        "- Có thể dùng Markdown cơ bản nếu hữu ích.\n"
        "\n---\n"
        f"{MATH_FORMAT_RULES}"
        "---\n\n"
    )

    if exercise_type == "mcq":
        return (
            MCQOutput,
            common_intro
            + "Hãy tạo 1 câu hỏi trắc nghiệm khách quan gồm đúng 4 đáp án A, B, C, D.\n"
            + "Yêu cầu:\n"
            + "- Có duy nhất 1 đáp án đúng.\n"
            + "- Distractors phải hợp lý và dễ gây nhầm lẫn với học sinh.\n"
            + "- Bloom 1-2 ưu tiên recall/understand; Bloom 3-6 tăng độ tình huống và lập luận.\n",
            serialize_exercise_result,
        )

    if exercise_type == "true_false":
        return (
            TrueFalseOutput,
            common_intro
            + "Hãy tạo 1 bài tập dạng Đúng/Sai.\n"
            + "Yêu cầu:\n"
            + "- `statement` là một mệnh đề duy nhất để học sinh đánh giá.\n"
            + "- `question` là lời dẫn ngắn gọn yêu cầu chọn True hoặc False.\n"
            + "- Mệnh đề phải kiểm tra kiến thức Bloom 1-2, tránh mơ hồ.\n",
            serialize_exercise_result,
        )

    if exercise_type == "fill_blank":
        return (
            FillBlankOutput,
            common_intro
            + "Hãy tạo 1 bài tập điền vào chỗ trống.\n"
            + "Yêu cầu:\n"
            + "- `sentence` phải chứa đúng 1 chỗ trống ký hiệu là `_____`.\n"
            + "- `blank_answers` liệt kê 1-3 đáp án chấp nhận được cho cùng chỗ trống.\n"
            + "- `hint` ngắn gọn, không lộ đáp án trực tiếp.\n"
            + "- Câu hỏi phù hợp Bloom 2-3, ưu tiên hiểu và áp dụng.\n",
            serialize_exercise_result,
        )

    if exercise_type == "multi_correct":
        return (
            MultiCorrectOutput,
            common_intro
            + "Hãy tạo 1 câu hỏi trắc nghiệm nhiều đáp án đúng gồm đúng 5 lựa chọn A, B, C, D, E.\n"
            + "Yêu cầu:\n"
            + "- Có từ 2 đến 4 đáp án đúng.\n"
            + "- Các lựa chọn sai phải gần đúng hoặc thiếu một điều kiện quan trọng.\n"
            + "- Câu hỏi phù hợp Bloom 4-5, đòi hỏi phân tích hoặc đánh giá.\n",
            serialize_exercise_result,
        )

    if exercise_type == "ordering":
        return (
            OrderingOutput,
            common_intro
            + "Hãy tạo 1 bài tập sắp xếp thứ tự.\n"
            + "Yêu cầu:\n"
            + "- `items` phải là danh sách các bước/ý/công đoạn ở thứ tự bị xáo trộn.\n"
            + "- `correct_order` phải chứa đúng các phần tử đó nhưng ở thứ tự đúng.\n"
            + "- Kiểm tra quy trình, lập luận tuần tự hoặc trình tự logic Bloom 3-4.\n",
            serialize_exercise_result,
        )

    if exercise_type == "matching":
        return (
            MatchingOutput,
            common_intro
            + "Hãy tạo 1 bài tập ghép nối khái niệm.\n"
            + "Yêu cầu:\n"
            + "- `pairs` gồm 3-5 cặp ghép đúng.\n"
            + "- `right_items` chứa đúng các giá trị cột phải nhưng ở thứ tự xáo trộn.\n"
            + "- Phù hợp Bloom 2-3, kiểm tra hiểu mối liên hệ giữa các khái niệm.\n",
            serialize_exercise_result,
        )

    return (
        ShortAnswerOutput,
        common_intro
        + "Hãy tạo 1 câu hỏi trả lời ngắn để chấm bằng rubric.\n"
        + "Yêu cầu:\n"
        + "- `question` là câu hỏi mở nhưng vẫn chấm được khách quan.\n"
        + "- `rubric` gồm 2-4 tiêu chí ngắn, rõ ràng.\n"
        + "- `sample_answer` là câu trả lời mẫu súc tích nhưng đầy đủ.\n"
        + "- Phù hợp Bloom 5-6, yêu cầu đánh giá hoặc sáng tạo ở mức ngắn gọn.\n",
        serialize_exercise_result,
    )


# ---------------------------------------------------------------------------
# Exercise generation
# ---------------------------------------------------------------------------
def generate_exercise(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    mastery: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Generate an exercise via LLM with a type selected from Bloom level and mastery."""
    bloom_label = BLOOM_VERBS.get(bloom_level, f"Level {bloom_level}")
    exercise_type = select_exercise_type(bloom_level, mastery)
    logger.info(
        f"[LLM-Gen] Concept: {concept_name} | Bloom: {bloom_label} | "
        f"Type: {exercise_type} | Mastery: {mastery}"
    )

    schema, prompt_base, serializer = _build_generation_spec(
        concept_name=concept_name,
        concept_definition=concept_definition,
        bloom_level=bloom_level,
        exercise_type=exercise_type,
    )
    structured_exercise_llm = get_structured_llm(schema)

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[LLM] ⏳ generate_exercise attempt {attempt}/{max_retries}")

            result = structured_exercise_llm.invoke(prompt_base)
            if not isinstance(result, schema):
                raise ValueError(f"LLM returned invalid type: {type(result)}")

            elapsed = time.time() - t0
            logger.info(f"[LLM] ✓ Exercise generated in {elapsed:.2f}s")
            return serializer(result)

        except Exception as e:
            logger.warning(
                f"[LLM] ⚠ generate_exercise attempt {attempt}/{max_retries} failed "
                f"(will_retry={attempt < max_retries}): {e}"
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)

    elapsed = time.time() - t0
    logger.error(f"[LLM] ✗ generate_exercise failed after {elapsed:.2f}s")
    raise RuntimeError("Exercise generation service is temporarily unavailable")


def evaluate_short_answer(
    *,
    concept_name: str,
    question: str,
    rubric: Sequence[str],
    sample_answer: str,
    student_answer: str,
) -> Dict[str, Any]:
    """Evaluate short-answer exercises against a rubric using structured LLM output."""
    logger.info(f"[LLM-Grade] Short answer grading for concept: {concept_name}")
    structured_grader_llm = get_structured_llm(ShortAnswerEvaluationOutput)
    rubric_lines = "\n".join(f"- {criterion}" for criterion in rubric)
    prompt = (
        "Bạn là giáo viên chấm câu trả lời ngắn theo rubric cố định.\n"
        f"Khái niệm: {concept_name}\n"
        f"Câu hỏi: {question}\n"
        f"Rubric:\n{rubric_lines}\n\n"
        f"Câu trả lời mẫu: {sample_answer}\n"
        f"Câu trả lời của học sinh: {student_answer}\n\n"
        "Yêu cầu chấm:\n"
        "- `is_correct` chỉ là true khi câu trả lời đáp ứng phần lớn rubric cốt lõi.\n"
        "- `score` từ 0-10.\n"
        "- `explanation` cần chỉ ra điểm đúng, điểm thiếu và gợi ý cải thiện.\n"
        "- Không phạt vì khác wording nếu ý đúng.\n"
        "- Trả lời bằng tiếng Việt.\n"
        "\n---\n"
        f"{MATH_FORMAT_RULES}"
        "---\n"
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[LLM] ⏳ evaluate_short_answer attempt {attempt}/{max_retries}")
            result = structured_grader_llm.invoke(prompt)
            if not isinstance(result, ShortAnswerEvaluationOutput):
                raise ValueError(f"LLM returned invalid type: {type(result)}")

            elapsed = time.time() - t0
            logger.info(f"[LLM] ✓ Short answer graded in {elapsed:.2f}s")
            return result.model_dump()
        except Exception as e:
            logger.warning(
                f"[LLM] ⚠ evaluate_short_answer attempt {attempt}/{max_retries} failed "
                f"(will_retry={attempt < max_retries}): {e}"
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)

    elapsed = time.time() - t0
    logger.error(f"[LLM] ✗ evaluate_short_answer failed after {elapsed:.2f}s")
    raise RuntimeError("Short-answer grading service is temporarily unavailable")


def generate_theory(
    concept_name: str,
    concept_definition: str,
) -> Optional[Dict[str, Any]]:
    """Generate a concise theory summary and examples via LLM using robust json schema."""
    logger.info(f"[LLM-Theory] Concept: {concept_name}")

    structured_theory_llm = get_structured_llm(TheoryOutput)

    prompt = (
        "Bạn là một giáo viên chuyên giải thích kiến thức một cách dễ hiểu, có hệ thống.\n"
        f"Hãy giải thích lý thuyết về khái niệm: {concept_name}\n"
        f"Định nghĩa gốc: {concept_definition}\n\n"
        "Yêu cầu:\n"
        "1. Phần 'content': Tóm tắt lý thuyết ngắn gọn (3-5 câu) theo cấu trúc:\n"
        "   - Câu 1: Định nghĩa khái niệm rõ ràng.\n"
        "   - Câu 2-3: Đặc điểm/tính chất quan trọng nhất.\n"
        "   - Câu 4-5 (nếu có): Công thức hoặc quy tắc chính.\n"
        "2. Phần 'examples': 2-3 ví dụ CỤ THỂ, mỗi ví dụ phải:\n"
        "   - Có số liệu/tình huống cụ thể (KHÔNG nói chung chung như 'ví dụ...' hay 'áp dụng...').\n"
        "   - Sắp xếp từ đơn giản đến phức tạp.\n"
        "   - Thể hiện cách áp dụng kiến thức vào bài toán thực.\n"
        "3. Ngôn ngữ: Tiếng Việt.\n"
        "4. Có thể dùng cú pháp Markdown cơ bản (*in nghiêng*, **in đậm**).\n"
        "\n---\n"
        f"{MATH_FORMAT_RULES}"
        "---\n"
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    fallback = {
        "content": f"Lý thuyết cơ bản về {concept_name}: {concept_definition}",
        "examples": ["Ví dụ 1: ...", "Ví dụ 2: ..."],
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[LLM] ⏳ generate_theory attempt {attempt}/{max_retries}")
            result = structured_theory_llm.invoke(prompt)

            if not isinstance(result, TheoryOutput):
                raise ValueError(f"LLM returned invalid type: {type(result)}")

            elapsed = time.time() - t0
            logger.info(f"[LLM] ✓ Theory generated in {elapsed:.2f}s")
            return result.model_dump()
        except Exception as e:
            logger.warning(
                f"[LLM] ⚠ generate_theory attempt {attempt}/{max_retries} failed "
                f"(will_retry={attempt < max_retries}): {e}"
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)

    elapsed = time.time() - t0
    logger.error(f"[LLM] ✗ generate_theory failed after {elapsed:.2f}s")
    return fallback
