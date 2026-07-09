from api.domains.learning.prompts import build_grading_messages, build_theory_messages


def test_grading_prompt_contains_score_anchors():
    messages = build_grading_messages(
        question="Q",
        rubric=["Ý 1", "Ý 2"],
        sample_answer="Mẫu",
        student_answer="HS",
    )

    assert "Thang điểm tham chiếu (0-10)" in messages[0].content
    assert "Không phạt vì khác wording" in messages[0].content


def test_theory_prompt_contains_example_constraint():
    messages = build_theory_messages(
        concept_name="Động năng",
        concept_definition="Động năng là năng lượng do chuyển động.",
    )

    assert "BÀI TOÁN CỤ THỂ" in messages[0].content
    assert "ví dụ cụ thể có lời giải ngắn" in messages[1].content
