from __future__ import annotations

from ..exercise_types import ExerciseType


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
        "- Distractor phải sai rõ ràng nhưng vẫn cùng loại khái niệm.\n"
    ),
    2: (
        "Bloom Level 2 - Understand:\n"
        "- Mục tiêu: kiểm tra học sinh có hiểu ý nghĩa, diễn giải lại được, phân biệt được mô tả đúng/sai của khái niệm.\n"
        "- Dạng yêu cầu phù hợp: giải thích ngắn, diễn đạt lại, chọn ví dụ/phản ví dụ đúng, nhận ra hệ quả trực tiếp từ định nghĩa.\n"
        "- KHÔNG chỉ yêu cầu chép lại định nghĩa nguyên văn.\n"
        "- Distractor nên dựa trên hiểu sai phổ biến.\n"
    ),
    3: (
        "Bloom Level 3 - Apply:\n"
        "- Mục tiêu: kiểm tra khả năng dùng quy tắc, công thức, định nghĩa đã học vào một bài toán hoặc tình huống cụ thể.\n"
        "- Dạng yêu cầu phù hợp: tính toán, thay số, thực hiện quy trình quen thuộc, áp dụng công thức trực tiếp.\n"
        "- Tình huống phải rõ ràng và đủ dữ kiện.\n"
    ),
    4: (
        "Bloom Level 4 - Analyze:\n"
        "- Mục tiêu: kiểm tra khả năng tách vấn đề thành phần nhỏ, so sánh, chỉ ra quan hệ, tìm lỗi hoặc phân loại.\n"
        "- Dạng yêu cầu phù hợp: so sánh hai trường hợp, xác định bước sai, tìm nguyên nhân, nhận ra cấu trúc hoặc mẫu hình.\n"
    ),
    5: (
        "Bloom Level 5 - Evaluate:\n"
        "- Mục tiêu: kiểm tra khả năng đưa ra nhận định có căn cứ, chọn phương án tốt hơn, đánh giá tính đúng/sai hoặc hợp lý.\n"
        "- Đáp án đúng phải dựa trên tiêu chí rõ ràng, không mơ hồ.\n"
    ),
    6: (
        "Bloom Level 6 - Create:\n"
        "- Mục tiêu: kiểm tra khả năng tạo lập cách giải, thiết kế ví dụ, xây dựng phương án hoặc tổng hợp ý tưởng mới.\n"
        "- Đáp án phải chấm được rõ ràng theo tiêu chí hoặc điều kiện cụ thể.\n"
    ),
}

EXERCISE_TYPE_BLOOM_GUIDANCE = {
    ExerciseType.MCQ: {
        1: "Với MCQ Bloom 1: ưu tiên nhận diện/ghi nhớ trực tiếp, tránh mẹo.",
        2: "Với MCQ Bloom 2: buộc học sinh hiểu và diễn giải bản chất, không chép lại định nghĩa.",
        3: "Với MCQ Bloom 3: thêm dữ kiện/tình huống ngắn để áp dụng công thức hoặc quy tắc.",
        4: "Với MCQ Bloom 4: các phương án nên đại diện cho các hướng phân tích khác nhau.",
        5: "Với MCQ Bloom 5: các phương án là các nhận định/lập luận cạnh tranh, chỉ một phương án có căn cứ tốt nhất.",
        6: "Với MCQ Bloom 6: chỉ dùng khi có thể đánh giá phương án tạo lập nào thỏa điều kiện tốt nhất.",
    },
    ExerciseType.TRUE_FALSE: {
        1: "Với True/False Bloom 1: mệnh đề kiểm tra fact hoặc định nghĩa cơ bản.",
        2: "Với True/False Bloom 2: mệnh đề phải phản ánh một diễn giải của khái niệm, không quá mơ hồ.",
    },
    ExerciseType.FILL_BLANK: {
        2: "Với Fill Blank Bloom 2: chỗ trống nên là từ/cụm ngắn thể hiện ý nghĩa hoặc hệ quả trực tiếp.",
        3: "Với Fill Blank Bloom 3: chỗ trống nên là kết quả áp dụng trực tiếp hoặc bước quan trọng.",
    },
    ExerciseType.MULTI_CORRECT: {
        3: "Với Multi Correct Bloom 3: nhiều đáp án đúng nên phản ánh nhiều trường hợp áp dụng đúng quy tắc.",
        4: "Với Multi Correct Bloom 4: người học phải phân tích từng lựa chọn, không chỉ nhận diện bề mặt.",
        5: "Với Multi Correct Bloom 5: các lựa chọn cần cân nhắc dựa trên tiêu chí đánh giá rõ ràng.",
    },
    ExerciseType.ORDERING: {
        3: "Với Ordering Bloom 3: thứ tự đúng phản ánh quy trình thao tác quen thuộc.",
        4: "Với Ordering Bloom 4: thứ tự đúng phản ánh logic phân tích hoặc quan hệ nguyên nhân-kết quả.",
    },
    ExerciseType.MATCHING: {
        2: "Với Matching Bloom 2: ghép khái niệm với ý nghĩa, ví dụ hoặc tính chất trực tiếp.",
        3: "Với Matching Bloom 3: ghép tình huống ngắn với quy tắc hoặc cách áp dụng phù hợp.",
    },
    ExerciseType.SHORT_ANSWER: {
        5: "Với Short Answer Bloom 5: rubric phải chấm được chất lượng nhận định và căn cứ lập luận.",
        6: "Với Short Answer Bloom 6: rubric phải chấm được tính đầy đủ, đúng điều kiện và hợp lý của phương án do học sinh tạo ra.",
    },
}

COMMON_RULES = (
    "Quy tắc chung BẮT BUỘC:\n"
    "- Ngôn ngữ: Tiếng Việt rõ ràng, tự nhiên.\n"
    "- CHỈ tạo bài tập hiển thị và làm được hoàn toàn bằng text.\n"
    "- KHÔNG yêu cầu xem hình ảnh, sơ đồ hoặc dữ kiện ngoài đề.\n"
    "- Mọi dữ kiện cần thiết phải nằm trong nội dung bài tập.\n"
)

NEGATIVE_CONSTRAINTS = {
    ExerciseType.MCQ: (
        "- KHÔNG dùng đáp án kiểu 'Tất cả đều đúng/sai'.\n"
        "- KHÔNG tạo 1 đáp án đúng quá dài hoặc quá nổi bật so với phần còn lại.\n"
        "- KHÔNG tạo distractor ngớ ngẩn hoặc vô lý hoàn toàn.\n"
    ),
    ExerciseType.TRUE_FALSE: (
        "- KHÔNG viết mệnh đề nước đôi, phụ thuộc diễn giải mơ hồ.\n"
        "- KHÔNG nhồi nhiều ý độc lập vào cùng một statement.\n"
    ),
    ExerciseType.FILL_BLANK: (
        "- KHÔNG tạo hơn 1 chỗ trống.\n"
        "- KHÔNG để chỗ trống có quá nhiều đáp án cùng đúng nhưng khác nghĩa.\n"
    ),
    ExerciseType.MULTI_CORRECT: (
        "- KHÔNG để số đáp án đúng là 0, 1 hoặc cả 5 đáp án.\n"
        "- KHÔNG làm lộ đáp án bằng mẫu hình chữ cái hoặc độ dài câu.\n"
    ),
    ExerciseType.ORDERING: (
        "- KHÔNG thêm bước thừa hoặc bước trùng nghĩa.\n"
        "- KHÔNG tạo các bước có thể đổi chỗ mà vẫn đúng.\n"
    ),
    ExerciseType.MATCHING: (
        "- KHÔNG tạo các cặp ghép mơ hồ, có thể ghép nhiều đáp án đều hợp lý.\n"
        "- KHÔNG dùng các mô tả gần như giống nhau cho nhiều right items.\n"
    ),
    ExerciseType.SHORT_ANSWER: (
        "- KHÔNG tạo câu hỏi quá mở khiến không thể chấm khách quan.\n"
        "- KHÔNG viết rubric mơ hồ hoặc trùng lặp ý.\n"
    ),
}

EXPLANATION_GUIDANCE = {
    ExerciseType.MCQ: (
        "- explanation_correct: giải thích vì sao phương án đúng là đúng.\n"
        "- explanation_incorrect: nêu lỗi hiểu sai phổ biến dẫn tới distractor.\n"
    ),
    ExerciseType.TRUE_FALSE: (
        "- explanation_correct: chỉ ra phần nào của statement làm nó đúng/sai.\n"
        "- explanation_incorrect: nói rõ điều kiện/phản ví dụ khiến học sinh dễ nhầm.\n"
    ),
    ExerciseType.FILL_BLANK: (
        "- explanation_correct: nêu ý nghĩa của cụm cần điền.\n"
        "- explanation_incorrect: nhắc điều kiện hoặc khái niệm mà người học hay nhầm.\n"
    ),
    ExerciseType.MULTI_CORRECT: (
        "- explanation_correct: giải thích logic chung của các lựa chọn đúng.\n"
        "- explanation_incorrect: nêu vì sao các lựa chọn sai thiếu điều kiện hoặc sai bản chất.\n"
    ),
    ExerciseType.ORDERING: (
        "- explanation_correct: giải thích vì sao trình tự này hợp lý.\n"
        "- explanation_incorrect: nói rõ bước nào cần đứng trước/sau và vì sao.\n"
    ),
    ExerciseType.MATCHING: (
        "- explanation_correct: nêu quan hệ đúng giữa các cặp chính.\n"
        "- explanation_incorrect: chỉ ra cặp nào dễ bị ghép sai và vì sao.\n"
    ),
    ExerciseType.SHORT_ANSWER: (
        "- explanation_correct: phản ánh đúng rubric và tiêu chí chấm.\n"
        "- explanation_incorrect: nhắc phần còn thiếu hoặc lập luận yếu theo rubric.\n"
    ),
}

EMPTY_DEFINITION_GUARD = (
    "Cảnh báo: định nghĩa kiến thức đang ngắn hoặc thiếu ngữ cảnh. Hãy suy luận thận trọng từ tên kiến thức, "
    "ưu tiên kiến thức nền tảng chuẩn, tránh bịa thêm chi tiết chuyên sâu không chắc chắn.\n"
)

EXPLANATION_TONE_GUIDANCE = (
    "Hướng dẫn giọng văn giải thích BẮT BUỘC:\n"
    "- explanation_correct: giọng tích cực, khích lệ. VD: 'Chính xác! ...', 'Rất tốt! ...'.\n"
    "- explanation_incorrect: giọng nhẹ nhàng, hướng dẫn. KHÔNG nói 'Sai rồi!' hay phê phán.\n"
    "  VD: 'Chưa chính xác. Hãy xem lại...', 'Lưu ý rằng...'.\n"
    "- Khi sai, gợi ý hướng ôn tập hoặc khái niệm cần xem lại.\n"
)

META_VALIDATION_CHECKLIST = (
    "Checklist tự kiểm tra trước khi trả lời:\n"
    "- [ ] Đã bám đúng Bloom level và mức phức tạp tương ứng.\n"
    "- [ ] Đã dùng LaTeX cho công thức toán.\n"
    "- [ ] Đề bài đủ dữ kiện và làm được hoàn toàn bằng text.\n"
    "- [ ] Các distractor/lựa chọn sai hợp lý, không lộ đáp án.\n"
    "- [ ] Explanation có giọng văn phù hợp (khích lệ / nhẹ nhàng).\n"
    "- [ ] Output khớp schema JSON yêu cầu.\n"
)

SCORE_ANCHORS = (
    "Thang điểm tham chiếu (0-10):\n"
    "- 9-10: Trả lời xuất sắc, đầy đủ ý, đúng bản chất và đáp ứng trọn rubric.\n"
    "- 7-8: Đúng hướng, đúng phần lớn rubric nhưng còn thiếu chi tiết nhỏ.\n"
    "- 5-6: Có ý đúng cơ bản nhưng còn thiếu ý quan trọng hoặc sai nhẹ.\n"
    "- 3-4: Lạc đề một phần hoặc sai phần lớn rubric.\n"
    "- 0-2: Sai hoàn toàn, gần như không đáp ứng rubric, hoặc bỏ trống.\n"
    "\n"
    "Ngưỡng pass/fail:\n"
    "- is_correct = True khi score >= 5 (học sinh nắm được ý cốt lõi).\n"
    "- is_correct = False khi score < 5 (thiếu hoặc sai phần lớn rubric).\n"
)

THEORY_EXAMPLES_CONSTRAINT = (
    "Ràng buộc ví dụ lý thuyết BẮT BUỘC:\n"
    "- Mỗi ví dụ phải là BÀI TOÁN CỤ THỂ hoặc tình huống cụ thể, không mô tả chung chung.\n"
    "- Mỗi ví dụ cần có lời giải hoặc diễn giải ngắn, rõ cách áp dụng.\n"
    "- Sắp xếp ví dụ từ cơ bản đến nâng dần.\n"
)
