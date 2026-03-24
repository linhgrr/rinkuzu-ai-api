from langchain_core.messages import HumanMessage, SystemMessage

from api.core.learning.exercise_types import ExerciseType
from api.core.learning.prompts import PromptBuilder


def test_build_system_message_contains_role_and_math_rules():
    builder = PromptBuilder(
        concept_name="Động năng",
        concept_definition="Động năng là năng lượng của vật do chuyển động mà có.",
        bloom_level=3,
        exercise_type=ExerciseType.MCQ,
    )

    system_message = builder.build_system_message()

    assert "Bạn là giáo viên chuyên tạo bài tập adaptive learning" in system_message
    assert "Quy tắc định dạng toán BẮT BUỘC" in system_message


def test_build_user_message_contains_concept_and_bloom():
    builder = PromptBuilder(
        concept_name="Động năng",
        concept_definition="Động năng là năng lượng của vật do chuyển động mà có.",
        bloom_level=3,
        exercise_type=ExerciseType.MCQ,
    )

    user_message = builder.build_user_message(
        type_spec="Hãy tạo MCQ.",
        negative_constraints="KHÔNG đoán mò.",
        recent_exercises=None,
    )

    assert "Kiến thức: Động năng" in user_message
    assert "Bloom level: 3" in user_message


def test_empty_definition_guard_added_when_short():
    builder = PromptBuilder(
        concept_name="Động năng",
        concept_definition="quá ngắn",
        bloom_level=2,
        exercise_type=ExerciseType.FILL_BLANK,
    )

    user_message = builder.build_user_message(
        type_spec="Spec",
        negative_constraints="KHÔNG mơ hồ.",
        recent_exercises=None,
    )

    assert "định nghĩa kiến thức đang ngắn hoặc thiếu ngữ cảnh" in user_message


def test_few_shot_included_for_each_type():
    for exercise_type in ExerciseType:
        builder = PromptBuilder(
            concept_name="Khái niệm",
            concept_definition="Đây là định nghĩa đủ dài để không kích hoạt guard.",
            bloom_level=2,
            exercise_type=exercise_type,
        )
        messages = builder.build_messages()
        human = next(message for message in messages if isinstance(message, HumanMessage))
        assert "Few-shot JSON mẫu" in human.content


def test_negative_constraints_included():
    builder = PromptBuilder(
        concept_name="Hình vuông",
        concept_definition="Hình vuông có bốn cạnh bằng nhau và bốn góc vuông.",
        bloom_level=2,
        exercise_type=ExerciseType.MCQ,
    )

    user_message = builder.build_messages()[1]

    assert "Ràng buộc negative constraints" in user_message.content


def test_meta_validation_checklist_present():
    builder = PromptBuilder(
        concept_name="Lực",
        concept_definition="Lực là đại lượng vectơ đặc trưng cho tác dụng của vật này lên vật khác.",
        bloom_level=3,
        exercise_type=ExerciseType.MCQ,
    )

    system_message, user_message = builder.build_messages()

    assert isinstance(system_message, SystemMessage)
    assert "Checklist tự kiểm tra trước khi trả lời" in user_message.content


def test_recent_exercises_dedup_block():
    builder = PromptBuilder(
        concept_name="Lực",
        concept_definition="Lực là đại lượng vectơ đặc trưng cho tác dụng của vật này lên vật khác.",
        bloom_level=3,
        exercise_type=ExerciseType.MCQ,
    )

    messages = builder.build_messages(
        recent_exercises=[
            {"question": "Q1", "exercise_type": "mcq", "bloom_level": 3},
        ]
    )

    assert "Các bài gần nhất cùng concept (TRÁNH trùng lặp với các bài dưới đây)" in messages[1].content
    assert "\"Q1\"" in messages[1].content
