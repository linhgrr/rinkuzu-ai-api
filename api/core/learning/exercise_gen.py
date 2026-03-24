"""
exercise_gen.py — LLM-powered exercise generation and answer evaluation.
"""

import json
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
from ..shared.llm import (
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


BLOOM_LEVEL_GUIDANCE = {
    1: (
        "Bloom Level 1 - Remember:\n"
        "- Mục tiêu: kiểm tra khả năng nhớ lại đúng sự kiện, định nghĩa, thuật ngữ, ký hiệu, quy tắc cơ bản.\n"
        "- Dạng yêu cầu phù hợp: nhận diện, chọn phát biểu đúng, nhắc lại định nghĩa, xác định tên gọi hoặc giá trị đã học.\n"
        "- Không yêu cầu giải thích sâu, suy luận nhiều bước hay áp dụng vào tình huống mới.\n"
        "- Distractor (đáp án sai): sai rõ ràng nhưng cùng loại khái niệm, tránh vô lý hoàn toàn.\n"
    ),
    2: (
        "Bloom Level 2 - Understand:\n"
        "- Mục tiêu: kiểm tra học sinh có hiểu ý nghĩa, diễn giải lại được, phân biệt được mô tả đúng/sai của khái niệm.\n"
        "- Dạng yêu cầu phù hợp: giải thích ngắn, diễn đạt lại, chọn ví dụ/phản ví dụ đúng, nhận ra hệ quả trực tiếp từ định nghĩa.\n"
        "- Có thể yêu cầu hiểu bản chất hoặc mô tả bằng lời, nhưng chưa nên đòi hỏi tính toán phức tạp hay quy trình nhiều bước.\n"
        "- KHÔNG được chỉ yêu cầu nhận diện lại định nghĩa nguyên văn.\n"
        "- Phải có yếu tố diễn giải, hiểu bản chất hoặc nhận ra quan hệ đơn giản.\n"
        "- Distractor: dựa trên các hiểu sai phổ biến (ví dụ: nhầm điều kiện, thiếu yếu tố, suy diễn sai nhẹ).\n"
    ),
    3: (
        "Bloom Level 3 - Apply:\n"
        "- Mục tiêu: kiểm tra khả năng dùng quy tắc, công thức, định nghĩa đã học vào một bài toán hoặc tình huống cụ thể.\n"
        "- Dạng yêu cầu phù hợp: tính toán, thay số, thực hiện quy trình quen thuộc, áp dụng công thức trực tiếp.\n"
        "- Tình huống nên rõ ràng và đủ dữ kiện; không biến thành phân tích sâu nhiều hướng.\n"
        "- Distractor: sai do áp dụng sai công thức, sai bước tính, hoặc nhầm dữ kiện.\n"
    ),
    4: (
        "Bloom Level 4 - Analyze:\n"
        "- Mục tiêu: kiểm tra khả năng tách vấn đề thành phần nhỏ, so sánh, chỉ ra quan hệ, tìm lỗi hoặc phân loại.\n"
        "- Dạng yêu cầu phù hợp: so sánh hai trường hợp, xác định bước sai, tìm nguyên nhân, nhận ra cấu trúc hoặc mẫu hình.\n"
        "- Cần có suy luận phân tích, không chỉ áp dụng công thức trực tiếp.\n"
        "- Distractor: sai do lập luận thiếu bước, nhầm quan hệ, hoặc phân tích chưa đầy đủ.\n"
    ),
    5: (
        "Bloom Level 5 - Evaluate:\n"
        "- Mục tiêu: kiểm tra khả năng đưa ra nhận định có căn cứ, chọn phương án tốt hơn, đánh giá tính đúng/sai hoặc hợp lý.\n"
        "- Dạng yêu cầu phù hợp: biện luận, phê bình lời giải, chọn lập luận thuyết phục nhất, đánh giá kết luận.\n"
        "- Đáp án đúng phải dựa trên tiêu chí rõ ràng, không mơ hồ.\n"
        "- Distractor: các phương án nghe hợp lý nhưng thiếu tiêu chí, lập luận yếu hoặc sai logic.\n"
    ),
    6: (
        "Bloom Level 6 - Create:\n"
        "- Mục tiêu: kiểm tra khả năng tạo lập cách giải, thiết kế ví dụ, xây dựng phương án hoặc tổng hợp ý tưởng mới.\n"
        "- Dạng yêu cầu phù hợp: đề xuất cách làm, xây dựng đáp án, tạo ví dụ thỏa điều kiện, hoàn thiện phương án.\n"
        "- Đáp án phải chấm được rõ ràng theo tiêu chí hoặc điều kiện cụ thể.\n"
        "- Distractor: phương án không thỏa điều kiện, thiếu tính đầy đủ hoặc vi phạm ràng buộc đề bài.\n"
    ),
}


EXERCISE_TYPE_BLOOM_GUIDANCE = {
    ExerciseType.MCQ: {
        1: "- Với MCQ Bloom 1: hỏi nhận diện/ghi nhớ trực tiếp; đáp án đúng nên là kiến thức cốt lõi, distractor sai gần nhưng không đánh đố.\n",
        2: "- Với MCQ Bloom 2: câu hỏi phải buộc học sinh hiểu và diễn giải bản chất; tránh chỉ chép lại nguyên văn định nghĩa.\n",
        3: "- Với MCQ Bloom 3: nên có tình huống ngắn hoặc dữ kiện cụ thể để học sinh áp dụng quy tắc/công thức.\n",
        4: "- Với MCQ Bloom 4: các phương án nên đại diện cho các cách phân tích/nhận định khác nhau, chỉ một phương án phân tích đúng.\n",
        5: "- Với MCQ Bloom 5: các phương án nên là các nhận định/lập luận cạnh tranh, đáp án đúng là nhận định có căn cứ tốt nhất.\n",
        6: "- Với MCQ Bloom 6: dùng khi có thể đánh giá phương án tạo lập/thiết kế nào thỏa điều kiện tốt nhất; không biến thành câu hỏi mẹo.\n",
    },
    ExerciseType.TRUE_FALSE: {
        1: "- Với True/False Bloom 1: mệnh đề kiểm tra nhớ đúng sai của một fact/định nghĩa cơ bản.\n",
        2: "- Với True/False Bloom 2: mệnh đề nên thể hiện một cách diễn giải của khái niệm để học sinh đánh giá mức độ hiểu đúng.\n",
    },
    ExerciseType.FILL_BLANK: {
        2: "- Với Fill Blank Bloom 2: chỗ trống nên là từ/cụm ngắn thể hiện ý nghĩa hoặc hệ quả trực tiếp từ khái niệm.\n",
        3: "- Với Fill Blank Bloom 3: chỗ trống nên là kết quả áp dụng trực tiếp hoặc bước quan trọng trong quy trình quen thuộc.\n",
    },
    ExerciseType.MULTI_CORRECT: {
        3: "- Với Multi Correct Bloom 3: nhiều đáp án đúng nên tương ứng nhiều trường hợp áp dụng đúng quy tắc.\n",
        4: "- Với Multi Correct Bloom 4: các lựa chọn nên buộc học sinh phân tích từng phương án thay vì nhận diện bề mặt.\n",
        5: "- Với Multi Correct Bloom 5: các lựa chọn nên là các nhận định cần cân nhắc theo tiêu chí rõ ràng trước khi chọn.\n",
    },
    ExerciseType.ORDERING: {
        3: "- Với Ordering Bloom 3: thứ tự đúng nên phản ánh quy trình áp dụng quen thuộc, có tính thao tác rõ ràng.\n",
        4: "- Với Ordering Bloom 4: thứ tự đúng nên phản ánh logic phân tích hoặc quan hệ nguyên nhân-kết quả giữa các bước.\n",
    },
    ExerciseType.MATCHING: {
        2: "- Với Matching Bloom 2: ghép khái niệm với ý nghĩa, ví dụ, tính chất hoặc hệ quả trực tiếp để kiểm tra hiểu quan hệ.\n",
        3: "- Với Matching Bloom 3: ghép tình huống/case ngắn với quy tắc hoặc cách áp dụng phù hợp.\n",
    },
    ExerciseType.SHORT_ANSWER: {
        5: "- Với Short Answer Bloom 5: rubric phải chấm được chất lượng nhận định và căn cứ lập luận.\n",
        6: "- Với Short Answer Bloom 6: rubric phải chấm được tính đầy đủ, đúng điều kiện và hợp lý của phương án do học sinh tạo ra.\n",
    },
}


def _build_generation_spec(
    concept_name: str,
    concept_definition: str,
    bloom_level: int,
    exercise_type: ExerciseType,
    recent_same_concept_exercises: Optional[Sequence[Dict[str, Any]]] = None,
):
    bloom_guidance = BLOOM_LEVEL_GUIDANCE.get(
        bloom_level,
        "Hãy bám sát mức độ tư duy Bloom được yêu cầu và tránh lệch sang cấp độ thấp hoặc cao hơn.\n",
    )
    exercise_type_guidance = EXERCISE_TYPE_BLOOM_GUIDANCE.get(exercise_type, {}).get(bloom_level, "")
    common_intro = (
        "Bạn là giáo viên chuyên tạo bài tập adaptive learning theo Bloom's Taxonomy.\n"
        f"Kiến thức yêu cầu: {concept_name}\n"
        f"Định nghĩa kiến thức: {concept_definition}\n"
        f"Mức độ tư duy (Bloom): Level {bloom_level} - {BLOOM_VERBS.get(bloom_level, '')}\n"
        f"{bloom_guidance}"
        f"{exercise_type_guidance}"
        "Ngôn ngữ: Tiếng Việt.\n"
        "Nội dung phải rõ ràng, không lan man, đúng trọng tâm kiến thức.\n"
        "BẮT BUỘC chỉ tạo bài tập có thể hiển thị và làm hoàn toàn bằng text.\n"
        "- Không được yêu cầu học sinh xem hình ảnh, hình vẽ, sơ đồ, đồ thị, biểu đồ, bảng, ký hiệu tô màu, hoặc bố cục trực quan.\n"
        "- Không được tự mô tả rằng có hình bên dưới, hình minh họa, ảnh đính kèm, hoặc 'quan sát hình sau'.\n"
        "- Không được sinh bài mà đáp án phụ thuộc vào yếu tố thị giác chưa được viết ra bằng text.\n"
        "- Nếu cần dữ kiện, hãy viết đầy đủ mọi dữ kiện trực tiếp trong câu hỏi bằng text hoặc Markdown đơn giản.\n"
        "- explanation_correct: 1-3 câu, giọng thân thiện, giải thích vì sao đáp án đúng.\n"
        "- explanation_incorrect: 1-2 câu, nêu lỗi sai phổ biến hoặc điều kiện cần còn thiếu.\n"
        "- Có thể dùng Markdown cơ bản nếu hữu ích.\n"
        "\n---\n"
        f"{MATH_FORMAT_RULES}"
        "---\n\n"
    )
    recent_examples_block = ""
    if recent_same_concept_exercises:
        serialized_recent = json.dumps(
            list(recent_same_concept_exercises),
            ensure_ascii=False,
            indent=2,
        )
        recent_examples_block = (
            "Các bài tập gần nhất của cùng khái niệm để tham chiếu tránh lặp:\n"
            f"{serialized_recent}\n"
            "Yêu cầu đa dạng hóa BẮT BUỘC:\n"
            "- Không lặp lại cùng ý tưởng câu hỏi, cùng dữ kiện chính, cùng bối cảnh, cùng ví dụ hoặc cùng đáp án đúng.\n"
            "- Nếu cùng dạng bài, phải đổi rõ rệt ngữ cảnh và cách kiểm tra kiến thức.\n"
            "- Hãy tạo bài mới kiểm tra cùng khái niệm nhưng khác nội dung so với danh sách trên.\n"
            "\n"
        )
    common_intro += recent_examples_block

    if exercise_type == ExerciseType.MCQ:
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

    if exercise_type == ExerciseType.TRUE_FALSE:
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

    if exercise_type == ExerciseType.FILL_BLANK:
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

    if exercise_type == ExerciseType.MULTI_CORRECT:
        return (
            MultiCorrectOutput,
            common_intro
            + "Hãy tạo 1 câu hỏi trắc nghiệm nhiều đáp án đúng gồm đúng 5 lựa chọn A, B, C, D, E.\n"
            + "Yêu cầu:\n"
            + "- Có từ 2 đến 4 đáp án đúng.\n"
            + "- Các lựa chọn sai phải gần đúng hoặc thiếu một điều kiện quan trọng.\n"
            + "- Câu hỏi phù hợp Bloom 3-5, đòi hỏi áp dụng, phân tích hoặc đánh giá.\n",
            serialize_exercise_result,
        )

    if exercise_type == ExerciseType.ORDERING:
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

    if exercise_type == ExerciseType.MATCHING:
        return (
            MatchingOutput,
            common_intro
            + "Hãy tạo 1 bài tập ghép nối khái niệm.\n"
            + "Yêu cầu:\n"
            + "- `pairs` gồm 3-5 cặp ghép đúng.\n"
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
    recent_same_concept_exercises: Optional[Sequence[Dict[str, Any]]] = None,
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
        recent_same_concept_exercises=recent_same_concept_exercises,
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
