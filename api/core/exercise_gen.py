"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.

Uses LangChain with an OpenAI-compatible local API endpoint.
- generate_exercise: Native JSON Schema structured output
- evaluate_answer:   Native JSON Schema structured output
- generate_theory:   Native JSON Schema structured output
"""

import json
import os
import time
from typing import Optional, Dict, Any, Literal, List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from loguru import logger

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
import sys
from pathlib import Path

# Add content-processor src to sys.path so we can reuse llm module
CONTENT_PROCESSOR_SRC = str(
    Path(__file__).resolve().parents[2] / "content-processor" / "src"
)
if CONTENT_PROCESSOR_SRC not in sys.path:
    sys.path.insert(0, CONTENT_PROCESSOR_SRC)

from llm import get_llm

_llm: Optional[ChatOpenAI] = None          # plain text invocation
_structured_exercise_llm = None            # with_structured_output for exercise
_structured_eval_llm = None                # with_structured_output for evaluation
_structured_theory_llm = None              # with_structured_output for theory


def init_llm(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Initialize ChatOpenAI pointing to an OpenAI-compatible endpoint."""
    global _llm, _structured_exercise_llm, _structured_eval_llm, _structured_theory_llm

    # get_llm already handles environment variables and normalization
    _llm = get_llm(temperature=0.3, base_url=base_url, model=model, api_key=api_key)

    print(f"[LLM] Connecting with model={_llm.model_name}")

    try:
        _structured_exercise_llm = _llm.with_structured_output(ExerciseOutput, method="json_schema")
        _structured_theory_llm = _llm.with_structured_output(TheoryOutput, method="json_schema")
    except Exception as e:
        print(f"[LLM] ⚠ Structured chain init failed: {e}")

    print("[LLM] ✓ Ready — structured chains initialized.")

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
    print(f"\n{'─'*60}")
    print("[LLM] ▶ generate_exercise called")
    print(f"  Concept  : {concept_name}")
    print(f"  Bloom    : Level {bloom_level} — {bloom_label}")
    print(f"  Def.     : {concept_definition[:120]}{'...' if len(concept_definition) > 120 else ''}")

    if _structured_exercise_llm is None:
        raise ValueError("[LLM] ⚠ LLM not initialized — generation failed")

    prompt_base = (
        "Bạn là một giáo viên xuất sắc và thân thiện.\n"
        "Hãy tạo 1 bài tập trắc nghiệm khách quan gồm đúng 4 đáp án A, B, C, D.\n"
        f"Kiến thức yêu cầu: {concept_name}\n"
        f"Định nghĩa kiến thức: {concept_definition}\n"
        f"Mức độ tư duy (Bloom): Level {bloom_level} - {BLOOM_VERBS.get(bloom_level, '')}\n\n"
        "Yêu cầu nội dung:\n"
        "- Câu hỏi rõ ràng, phù hợp đúng Bloom level.\n"
        "- Có duy nhất 1 đáp án đúng.\n"
        "- Giải thích chi tiết cho phương án đúng (explanation_correct) và gợi ý sửa sai cho phương án sai (explanation_incorrect).\n"
        "- Nếu có công thức toán học, BẮT BUỘC bỏ trong cặp dấu $...$ (ví dụ: $x^2 + y^2 = z^2$).\n"
        "- Có thể có xuống dòng, định dạng in đậm/nghiêng bằng Markdown nếu cần thiết.\n"
    )

    t0 = time.time()
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[LLM] ⏳ generate_exercise attempt {attempt}/{max_retries}")
            
            result = _structured_exercise_llm.invoke(prompt_base)
            if not isinstance(result, ExerciseOutput):
                raise ValueError(f"LLM returned invalid type: {type(result)}")

            elapsed = time.time() - t0
            print(f"[LLM] ✓ Exercise generated in {elapsed:.2f}s")
            print(f"  Q: {result.question[:120]}{'...' if len(result.question) > 120 else ''}")
            print(f"  Correct: {result.correct_option}")
            print(f"{'─'*60}")
            return _exercise_to_dict(result)

        except Exception as e:
            print(f"[LLM] ⚠ generate_exercise attempt {attempt} failed: {e}")

    elapsed = time.time() - t0
    print(f"[LLM] ✗ generate_exercise failed after {elapsed:.2f}s")
    print(f"{'─'*60}")
    raise RuntimeError("Failed to generate exercise after max retries")


def generate_theory(
    concept_name: str,
    concept_definition: str,
) -> Optional[Dict[str, Any]]:
    """Generate a concise theory summary and examples via LLM using robust json schema."""
    print(f"\n{'─'*60}")
    print("[LLM] ▶ generate_theory called")
    print(f"  Concept: {concept_name}")

    if _structured_theory_llm is None:
        return {
            "content": f"Lý thuyết về {concept_name}.",
            "examples": [f"Ví dụ về {concept_name} 1", f"Ví dụ về {concept_name} 2"]
        }

    prompt = (
        "Bạn là một giáo viên xuất sắc.\n"
        f"Hãy giải thích lý thuyết về khái niệm: {concept_name}\n"
        f"Định nghĩa gốc: {concept_definition}\n\n"
        "Yêu cầu:\n"
        "1. Phần 'content': Tóm tắt lý thuyết cực kỳ ngắn gọn, súc tích (khoảng 3-5 câu), tập trung vào ý chính dễ hiểu.\n"
        "2. Phần 'examples': Đưa ra 2-3 ví dụ minh họa thực tế hoặc bài toán đơn giản.\n"
        "3. Ngôn ngữ: Tiếng Việt.\n"
        "4. Nếu có công thức toán học, BẮT BUỘC bỏ trong cặp dấu $...$ (ví dụ: $E = mc^2$).\n"
        "5. Có thể dùng cú pháp Markdown cơ bản (*in nghiêng*, **in đậm**).\n"
    )

    t0 = time.time()
    try:
        print(f"[LLM] ⏳ generate_theory attempt")
        result = _structured_theory_llm.invoke(prompt)
        
        if not isinstance(result, TheoryOutput):
            raise ValueError(f"LLM returned invalid type: {type(result)}")

        elapsed = time.time() - t0
        print(f"[LLM] ✓ Theory generated in {elapsed:.2f}s")
        print(f"{'─'*60}")
        return result.model_dump()
        
    except Exception as e:
        print(f"[LLM] ✗ generate_theory failed: {e}")
        return {
            "content": f"Lý thuyết cơ bản về {concept_name}: {concept_definition}",
            "examples": ["Ví dụ 1: ...", "Ví dụ 2: ..."]
        }


