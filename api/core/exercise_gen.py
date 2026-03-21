"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.
"""

import time
from typing import Optional, Dict, Any, Literal, List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from loguru import logger

from ..config import get_settings

# ---------------------------------------------------------------------------
# Bloom's Taxonomy labels
# ---------------------------------------------------------------------------
BLOOM_VERBS = {
    1: "Remember (Nho: Dinh nghia, liet ke, ghi nho)",
    2: "Understand (Hieu: Giai thich, tom tat)",
    3: "Apply (Van dung: Tinh toan, ap dung cong thuc)",
    4: "Analyze (Phan tich: So sanh, doi chieu, chia nho van de)",
    5: "Evaluate (Danh gia: Bien luan, phan xet tinh dung sai)",
    6: "Create (Sang tao: Thiet ke, chung minh, tong hop)",
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


# ---------------------------------------------------------------------------
# Global LLM instances
# ---------------------------------------------------------------------------
from .content_pipeline.infrastructure.runtime import get_content_processor_llm_factory

_llm: Optional[ChatOpenAI] = None          # plain text invocation
_structured_exercise_llm = None            # with_structured_output for exercise
_structured_theory_llm = None              # with_structured_output for theory

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


def _resolve_retry_policy() -> tuple[int, float]:
    settings = get_settings()
    return (
        max(1, int(settings.adaptive_llm_retry_attempts)),
        max(0.0, float(settings.adaptive_llm_retry_backoff_sec)),
    )


def _resolve_exercise_llm_model(explicit_model: Optional[str]) -> Optional[str]:
    settings = get_settings()
    return settings.adaptive_exercise_llm_model or explicit_model


def _sleep_before_retry(attempt: int, base_delay_sec: float) -> None:
    if base_delay_sec <= 0:
        return
    time.sleep(base_delay_sec * attempt)


def init_llm(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Initialize ChatOpenAI pointing to an OpenAI-compatible endpoint."""
    global _llm, _structured_exercise_llm, _structured_theory_llm

    get_llm = get_content_processor_llm_factory()
    selected_model = _resolve_exercise_llm_model(model)
    _llm = get_llm(
        temperature=0.3,
        base_url=base_url,
        model=selected_model,
        api_key=api_key,
    )

    logger.info(f"[LLM] Connecting with model={_llm.model_name}")

    try:
        _structured_exercise_llm = _llm.with_structured_output(ExerciseOutput, method="json_schema")
        _structured_theory_llm = _llm.with_structured_output(TheoryOutput, method="json_schema")
    except Exception as e:
        logger.warning(f"[LLM] ⚠ Structured chain init failed: {e}")

    logger.info("[LLM] ✓ Ready — structured chains initialized.")


def init_gemini(api_key: Optional[str] = None):
    """Backward-compatible wrapper — delegates to init_llm."""
    init_llm(api_key=api_key)


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

    if _structured_exercise_llm is None:
        raise ValueError("[LLM] ⚠ LLM not initialized — generation failed")

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
        "- Giải thích chi tiết cho phương án đúng (explanation_correct) và gợi ý sửa sai cho phương án sai (explanation_incorrect).\n"
        f"{MATH_FORMAT_RULES}"
        "- Có thể có xuống dòng, định dạng in đậm/nghiêng bằng Markdown nếu cần thiết.\n"
    )

    t0 = time.time()
    max_retries, backoff_sec = _resolve_retry_policy()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[LLM] ⏳ generate_exercise attempt {attempt}/{max_retries}")
            
            result = _structured_exercise_llm.invoke(prompt_base)
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
                _sleep_before_retry(attempt, backoff_sec)

    elapsed = time.time() - t0
    logger.error(f"[LLM] ✗ generate_exercise failed after {elapsed:.2f}s")
    raise RuntimeError("Exercise generation service is temporarily unavailable")


def generate_theory(
    concept_name: str,
    concept_definition: str,
) -> Optional[Dict[str, Any]]:
    """Generate a concise theory summary and examples via LLM using robust json schema."""
    logger.info(f"[LLM-Theory] Concept: {concept_name}")

    if _structured_theory_llm is None:
        return {
            "content": f"Lý thuyết về {concept_name}.",
            "examples": [f"Ví dụ về {concept_name} 1", f"Ví dụ về {concept_name} 2"]
        }

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
        f"4. {MATH_FORMAT_RULES}"
        "5. Có thể dùng cú pháp Markdown cơ bản (*in nghiêng*, **in đậm**).\n"
    )

    t0 = time.time()
    max_retries, backoff_sec = _resolve_retry_policy()
    fallback = {
        "content": f"Lý thuyết cơ bản về {concept_name}: {concept_definition}",
        "examples": ["Ví dụ 1: ...", "Ví dụ 2: ..."],
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[LLM] ⏳ generate_theory attempt {attempt}/{max_retries}")
            result = _structured_theory_llm.invoke(prompt)

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
                _sleep_before_retry(attempt, backoff_sec)

    elapsed = time.time() - t0
    logger.error(f"[LLM] ✗ generate_theory failed after {elapsed:.2f}s")
    return fallback
