"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.
"""

import time
from typing import Optional, Dict, Any, Literal, List

from pydantic import BaseModel, Field
from loguru import logger

from .llm import (
    get_structured_llm,
    resolve_retry_policy,
    sleep_before_retry,
)

# ---------------------------------------------------------------------------
# Bloom's Taxonomy labels
# ---------------------------------------------------------------------------
BLOOM_VERBS = {
    1: "Remember (Nhớ: Định nghĩa, liệt kê, ghi nhớ)",
    2: "Understand (Hiểu: Giải thích, tóm tắt)",
    3: "Apply (Vận dụng: Tính toán, áp dụng công thức)",
    4: "Analyze (Phân tích: So sánh, đối chiếu, chia nhỏ vấn đề)",
    5: "Evaluate (Đánh giá: Biện luận, phán xét tính đúng sai)",
    6: "Create (Sáng tạo: Thiết kế, chứng minh, tổng hợp)",
}


# ---------------------------------------------------------------------------
# Pydantic schema for structured exercise output
# ---------------------------------------------------------------------------
class ExerciseOptions(BaseModel):
    A: str = Field(..., description="Option A")
    B: str = Field(..., description="Option B")
    C: str = Field(..., description="Option C")
    D: str = Field(..., description="Option D")


class ExerciseOutput(BaseModel):
    """Multiple-choice exercise payload."""
    question: str = Field(..., description="Question text")
    options: ExerciseOptions = Field(..., description="Four options A/B/C/D")
    correct_option: Literal["A", "B", "C", "D"] = Field(..., description="Correct option label")
    explanation_correct: str = Field(..., description="Short friendly explanation for the correct answer")
    explanation_incorrect: str = Field(..., description="Short friendly explanation for incorrect answers")


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
def _exercise_to_dict(result: ExerciseOutput) -> Dict[str, Any]:
    return {
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


# ---------------------------------------------------------------------------
# Exercise generation
# ---------------------------------------------------------------------------
def generate_exercise(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
) -> Optional[Dict[str, Any]]:
    """Generate a multiple-choice exercise via LLM with strict json_schema struct output."""
    bloom_label = BLOOM_VERBS.get(bloom_level, f"Level {bloom_level}")
    
    logger.info(f"[LLM-Gen] Concept: {concept_name} | Bloom: {bloom_label}")

    structured_exercise_llm = get_structured_llm(ExerciseOutput)

    bloom_guidelines = {
        1: (
            "Hướng dẫn Bloom 1 (Nhớ): Hỏi về định nghĩa, công thức, thuật ngữ hoặc liệt kê. "
            "Đáp án sai phải là các thuật ngữ/định nghĩa gần giống, dễ nhầm lẫn.\n"
        ),
        2: (
            "Hướng dẫn Bloom 2 (Hiểu): Hỏi giải thích ý nghĩa, diễn giải hoặc suy luận từ định nghĩa. "
            "KHÔNG hỏi nguyên văn định nghĩa. Đáp án sai nên là các hiểu lầm phổ biến.\n"
        ),
        3: (
            "Hướng dẫn Bloom 3 (Áp dụng): Cho một bài toán/tình huống CỤ THỂ yêu cầu tính toán hoặc áp dụng công thức. "
            "Đáp án sai NÊN là kết quả khi tính sai ở một bước phổ biến (như nhầm dấu, quên đổi đơn vị).\n"
        ),
        4: (
            "Hướng dẫn Bloom 4 (Phân tích): Hỏi so sánh, phân biệt, tìm mối quan hệ giữa các khái niệm. "
            "Câu hỏi cần phân tích nhiều khía cạnh, không chỉ hỏi đúng/sai đơn giản.\n"
        ),
        5: (
            "Hướng dẫn Bloom 5 (Đánh giá): Hỏi đánh giá tính đúng/sai của mệnh đề, chọn giải pháp tối ưu, "
            "hoặc biện luận. Đáp án sai nên là các mệnh đề gần đúng nhưng có lỗi logic tinh vi.\n"
        ),
        6: (
            "Hướng dẫn Bloom 6 (Sáng tạo): Hỏi thiết kế phương pháp, tổng hợp kiến thức, hoặc đề xuất giải pháp mới. "
            "Yêu cầu học sinh vận dụng sáng tạo, không chỉ áp dụng công thức có sẵn.\n"
        ),
    }

    prompt_base = (
        "Bạn là một giáo viên chuyên soạn đề thi trắc nghiệm theo thang Bloom's Taxonomy.\n"
        "Hãy tạo 1 bài tập trắc nghiệm khách quan gồm đúng 4 đáp án A, B, C, D.\n\n"
        f"Kiến thức yêu cầu: {concept_name}\n"
        f"Định nghĩa kiến thức: {concept_definition}\n"
        f"Mức độ tư duy (Bloom): Level {bloom_level} - {BLOOM_VERBS.get(bloom_level, '')}\n"
        f"{bloom_guidelines.get(bloom_level, '')}\n"
        "Yêu cầu nội dung:\n"
        "- Câu hỏi rõ ràng, phù hợp ĐÚNG Bloom level yêu cầu ở trên.\n"
        "- Có duy nhất 1 đáp án đúng.\n"
        "- Yêu cầu chất lượng đáp án sai (distractors):\n"
        "  + Mỗi đáp án sai phải có lý do hợp lý mà học sinh hay nhầm.\n"
        "  + Không được có đáp án sai quá hiển nhiên hoặc vô nghĩa.\n"
        "  + Đáp án sai nên bắt nguồn từ lỗi khái niệm hoặc tính toán phổ biến.\n"
        "- explanation_correct: 1-3 câu, giọng thân thiện, có công thức LaTeX nếu cần.\n"
        "- explanation_incorrect: 1-2 câu, chỉ ra lỗi sai điển hình mà học sinh hay mắc.\n"
        "- Có thể có xuống dòng, định dạng in đậm/nghiêng bằng Markdown nếu cần thiết.\n"
        "\n---\n"
        f"{MATH_FORMAT_RULES}"
        "---\n\n"
        "VÍ DỤ CÂU HỎI TỐT (Bloom 3 - Áp dụng):\n"
        "question: Vật có khối lượng $m = 2$ kg chuyển động với vận tốc $v = 3$ m/s. "
        "Động năng của vật là bao nhiêu?\n"
        "options: A. $6$ J  B. $9$ J  C. $12$ J  D. $18$ J\n"
        "correct_option: B\n"
        "explanation_correct: Áp dụng $W_đ = \\frac{1}{2}mv^2 = \\frac{1}{2} \\cdot 2 \\cdot 3^2 = 9$ J.\n"
        "explanation_incorrect: Lỗi thường gặp: quên bình phương $v$ (được $6$ J) hoặc quên hệ số $\\frac{1}{2}$ (được $18$ J).\n"
    )

    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[LLM] ⏳ generate_exercise attempt {attempt}/{max_retries}")
            
            result = structured_exercise_llm.invoke(prompt_base)
            if not isinstance(result, ExerciseOutput):
                raise ValueError(f"LLM returned invalid type: {type(result)}")

            elapsed = time.time() - t0
            logger.info(f"[LLM] ✓ Exercise generated in {elapsed:.2f}s")
            return _exercise_to_dict(result)

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
