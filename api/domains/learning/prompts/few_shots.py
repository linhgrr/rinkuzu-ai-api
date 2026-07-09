from __future__ import annotations

from typing import Any

from api.domains.learning.exercise_types import ExerciseType

# Primary few-shot examples (shown by default) — covers both STEM and non-STEM.
FEW_SHOT_EXAMPLES: dict[ExerciseType, dict[str, Any]] = {
    ExerciseType.MCQ: {
        "exercise_type": "mcq",
        "question": "Đại lượng nào sau đây là đại lượng vô hướng?",
        "options": {
            "A": "Vận tốc",
            "B": "Gia tốc",
            "C": "Tốc độ",
            "D": "Lực",
        },
        "correct_option": "C",
        "explanation_correct": "Chính xác! Tốc độ chỉ có độ lớn, không có hướng nên là đại lượng vô hướng.",
        "explanation_incorrect": "Chưa chính xác. Hãy xem lại: vận tốc, gia tốc và lực đều có cả độ lớn lẫn hướng nên là đại lượng vectơ.",
    },
    ExerciseType.TRUE_FALSE: {
        "exercise_type": "true_false",
        "question": "Đánh giá phát biểu sau là đúng hay sai.",
        "statement": "Số 2 là số nguyên tố chẵn duy nhất.",
        "correct_answer": True,
        "explanation_correct": "Rất tốt! Số 2 chỉ có hai ước dương là 1 và 2, đồng thời là số chẵn duy nhất thỏa điều kiện đó.",
        "explanation_incorrect": "Chưa chính xác. Lưu ý rằng số 2 là ngoại lệ duy nhất — hãy xem lại định nghĩa số nguyên tố.",
    },
    ExerciseType.FILL_BLANK: {
        "exercise_type": "fill_blank",
        "question": "Điền vào chỗ trống.",
        "sentence": "Hình có bốn cạnh bằng nhau và bốn góc vuông là hình _____.",
        "blank_answers": ["vuông"],
        "hint": "Đó là trường hợp đặc biệt của hình chữ nhật và hình thoi.",
        "explanation_correct": "Chính xác! Đó là hình vuông.",
        "explanation_incorrect": "Chưa chính xác. Hãy nhớ: cần đồng thời có bốn cạnh bằng nhau và bốn góc vuông thì mới là hình vuông.",
    },
    ExerciseType.MULTI_CORRECT: {
        "exercise_type": "multi_correct",
        "question": "Trong các phát biểu sau, phát biểu nào đúng về hàm số bậc nhất?",
        "options": {
            "A": "Đồ thị là một đường thẳng",
            "B": "Có dạng $y=ax+b$ với $a \\neq 0$",
            "C": "Luôn đi qua gốc tọa độ",
            "D": "Khi $a>0$ thì hàm số đồng biến",
            "E": "Luôn có hai nghiệm",
        },
        "correct_options": ["A", "B", "D"],
        "explanation_correct": "Rất tốt! Ba lựa chọn này mô tả đúng tính chất cơ bản của hàm số bậc nhất.",
        "explanation_incorrect": "Chưa chính xác. Lưu ý: hàm số bậc nhất không nhất thiết đi qua gốc tọa độ (trừ khi $b=0$) và cũng không phải 'có hai nghiệm'.",
    },
    ExerciseType.ORDERING: {
        "exercise_type": "ordering",
        "question": "Sắp xếp đúng các bước giải phương trình bậc hai.",
        "items": [
            "Tính $\\Delta=b^2-4ac$",
            "Kết luận nghiệm của phương trình",
            "Xét dấu của $\\Delta$",
            "Áp dụng công thức nghiệm phù hợp",
        ],
        "correct_order": [
            "Tính $\\Delta=b^2-4ac$",
            "Xét dấu của $\\Delta$",
            "Áp dụng công thức nghiệm phù hợp",
            "Kết luận nghiệm của phương trình",
        ],
        "explanation_correct": "Chính xác! Đây là trình tự chuẩn để tránh bỏ sót trường hợp của phương trình bậc hai.",
        "explanation_incorrect": "Chưa đúng thứ tự. Hãy nhớ: không thể áp dụng công thức nghiệm trước khi xét dấu của $\\Delta$.",
    },
    ExerciseType.MATCHING: {
        "exercise_type": "matching",
        "question": "Ghép đại lượng với đơn vị tương ứng.",
        "pairs": [
            {"left": "Lực", "right": "Niutơn (N)"},
            {"left": "Công", "right": "Jun (J)"},
            {"left": "Công suất", "right": "Oát (W)"},
        ],
        "explanation_correct": "Chính xác! Mỗi đại lượng đã được ghép đúng với đơn vị SI tương ứng.",
        "explanation_incorrect": "Chưa chính xác. Hãy ôn lại các đơn vị SI: lực tính bằng Niutơn (N), công tính bằng Jun (J), công suất tính bằng Oát (W).",
    },
    ExerciseType.SHORT_ANSWER: {
        "exercise_type": "short_answer",
        "question": "Vì sao doanh nghiệp trong thị trường cạnh tranh hoàn hảo không thể tự ý tăng giá?",
        "rubric": [
            "Nêu rõ doanh nghiệp là người chấp nhận giá.",
            "Giải thích do sản phẩm đồng nhất và người mua dễ so sánh.",
            "Nêu hậu quả: tăng giá sẽ mất khách hàng.",
        ],
        "sample_answer": "Doanh nghiệp là người chấp nhận giá vì sản phẩm đồng nhất và người mua có thể chuyển sang đối thủ ngay. Nếu tăng giá, doanh nghiệp sẽ mất khách hàng.",
        "explanation_correct": "Rất tốt! Câu trả lời tốt cần nêu bản chất price taker và hệ quả khi tăng giá.",
        "explanation_incorrect": "Chưa đầy đủ. Hãy xem lại: nếu chỉ nói 'do cạnh tranh' mà không giải thích cơ chế mất khách thì chưa đủ rubric.",
    },
}

# Additional few-shot examples for higher Bloom levels (4-6).
# These are appended alongside the primary example when bloom_level >= 4.
FEW_SHOT_HIGH_BLOOM: dict[ExerciseType, dict[str, Any]] = {
    ExerciseType.MCQ: {
        "exercise_type": "mcq",
        "question": "Một học sinh giải phương trình $2x^2 - 3x + 1 = 0$ và được nghiệm $x = 1$ và $x = 2$. Sai lầm của học sinh nằm ở bước nào?",
        "options": {
            "A": "Tính sai $\\Delta$",
            "B": "Áp dụng sai công thức nghiệm",
            "C": "Tính đúng $\\Delta$ nhưng sai khi chia cho $2a$",
            "D": "Phân tích nhân tử sai",
        },
        "correct_option": "C",
        "explanation_correct": "Chính xác! Nghiệm đúng là $x=1$ và $x=\\frac{1}{2}$, cho thấy học sinh chia sai ở bước cuối.",
        "explanation_incorrect": "Chưa chính xác. Hãy tự thay nghiệm $x=2$ vào phương trình để kiểm tra — nó không thỏa. Lưu ý bước chia cho $2a$ là nơi dễ sai.",
    },
    ExerciseType.MULTI_CORRECT: {
        "exercise_type": "multi_correct",
        "question": "Xét hai hàm số $f(x) = x^2$ và $g(x) = |x|$ trên $\\mathbb{R}$. Những nhận định nào sau đây đúng?",
        "options": {
            "A": "Cả hai đều là hàm chẵn",
            "B": "Cả hai đều liên tục trên $\\mathbb{R}$",
            "C": "Cả hai đều khả vi tại $x=0$",
            "D": "$f(x) \\geq g(x)$ với mọi $x$",
            "E": "Cả hai đều đồng biến trên $(0, +\\infty)$",
        },
        "correct_options": ["A", "B", "E"],
        "explanation_correct": "Rất tốt! Cả hai đều chẵn, liên tục, và đồng biến trên $(0, +\\infty)$. Lưu ý $g(x)=|x|$ không khả vi tại $x=0$.",
        "explanation_incorrect": "Hãy kiểm tra từng nhận định: $|x|$ không khả vi tại $x=0$ (C sai), và $|x| > x^2$ khi $0 < x < 1$ (D sai).",
    },
    ExerciseType.SHORT_ANSWER: {
        "exercise_type": "short_answer",
        "question": "So sánh ưu và nhược điểm của năng lượng mặt trời và năng lượng gió, từ đó đề xuất giải pháp phối hợp hai nguồn năng lượng này.",
        "rubric": [
            "Nêu ít nhất 1 ưu và 1 nhược của mỗi nguồn năng lượng.",
            "So sánh yếu tố phụ thuộc thời tiết/mùa.",
            "Đề xuất hệ thống hybrid khả thi với lý do.",
        ],
        "sample_answer": "Năng lượng mặt trời mạnh vào ban ngày nhưng yếu khi trời mưa; năng lượng gió hoạt động cả ngày đêm nhưng phụ thuộc tốc độ gió. Phối hợp cả hai tạo hệ thống hybrid ổn định hơn vì khi một nguồn yếu, nguồn kia có thể bù.",
        "explanation_correct": "Rất tốt! Bạn đã phân tích đầy đủ và đưa ra giải pháp phối hợp hợp lý.",
        "explanation_incorrect": "Chưa đầy đủ. Hãy đảm bảo so sánh cụ thể cả ưu-nhược của MỖI nguồn, rồi mới đề xuất giải pháp.",
    },
}

# Additional few-shot examples for non-STEM subjects.
# Appended alongside the primary example when concept_name suggests a non-STEM subject.
FEW_SHOT_NON_STEM: dict[ExerciseType, dict[str, Any]] = {
    ExerciseType.MCQ: {
        "exercise_type": "mcq",
        "question": "Tác phẩm 'Chí Phèo' của Nam Cao thuộc trào lưu văn học nào?",
        "options": {
            "A": "Văn học lãng mạn",
            "B": "Văn học hiện thực phê phán",
            "C": "Văn học cách mạng",
            "D": "Văn học trung đại",
        },
        "correct_option": "B",
        "explanation_correct": "Chính xác! 'Chí Phèo' phản ánh hiện thực nông thôn Việt Nam trước 1945 với cái nhìn phê phán xã hội.",
        "explanation_incorrect": "Chưa chính xác. Hãy ôn lại: trào lưu hiện thực phê phán tập trung vào việc phơi bày mâu thuẫn xã hội, khác với văn học lãng mạn (tình cảm, lý tưởng).",
    },
    ExerciseType.TRUE_FALSE: {
        "exercise_type": "true_false",
        "question": "Đánh giá phát biểu sau là đúng hay sai.",
        "statement": "Cách mạng tháng Tám năm 1945 diễn ra thành công nhờ sự kết hợp giữa lực lượng chính trị và lực lượng vũ trang.",
        "correct_answer": True,
        "explanation_correct": "Rất tốt! Đây là nhận định đúng — Cách mạng tháng Tám thành công nhờ kết hợp đấu tranh chính trị và khởi nghĩa vũ trang.",
        "explanation_incorrect": "Chưa chính xác. Hãy xem lại: Cách mạng tháng Tám sử dụng cả hai lực lượng chứ không chỉ một.",
    },
    ExerciseType.SHORT_ANSWER: {
        "exercise_type": "short_answer",
        "question": "Phân tích ý nghĩa hình ảnh 'ánh trăng' trong bài thơ 'Ánh trăng' của Nguyễn Duy.",
        "rubric": [
            "Nêu hình ảnh ánh trăng tượng trưng cho quá khứ/thiên nhiên/đồng đội.",
            "Phân tích sự đối lập giữa ánh trăng trước và sau khi vào thành phố.",
            "Rút ra bài học về lòng thủy chung, uống nước nhớ nguồn.",
        ],
        "sample_answer": "Ánh trăng tượng trưng cho quá khứ gian khổ, nghĩa tình. Khi vào thành phố, nhân vật quên trăng — tức quên quá khứ. Ánh trăng 'im phăng phắc' cuối bài thức tỉnh lương tâm, nhắc nhở lòng thủy chung.",
        "explanation_correct": "Rất tốt! Bạn đã nắm được ý nghĩa biểu tượng và thông điệp tác phẩm.",
        "explanation_incorrect": "Chưa đủ sâu. Hãy phân tích thêm sự đối lập trước/sau khi vào thành phố và ý nghĩa hình ảnh 'im phăng phắc'.",
    },
}
