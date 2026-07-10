import asyncio
from types import SimpleNamespace

import pytest

from api.domains.quiz import quiz_tutor, tutor_chat
from api.shared.llm import normalize_chat_messages


def test_generate_quiz_tutor_response_uses_project_standard_message_shape(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_generate_tutor_text(*, input_messages, model, timeout_sec, action, max_tokens):
        captured["model"] = model
        captured["input_messages"] = input_messages
        captured["timeout_sec"] = timeout_sec
        captured["action"] = action
        captured["max_tokens"] = max_tokens
        return "Đây là phần giải thích đủ dài cho học sinh hiểu bài."

    monkeypatch.setattr(tutor_chat, "generate_tutor_text", fake_generate_tutor_text)
    monkeypatch.setattr(tutor_chat, "_resolve_tutor_model", lambda: "shared-model")
    monkeypatch.setattr(
        tutor_chat,
        "get_settings",
        lambda: SimpleNamespace(
            llm_timeout_sec=5,
            exercise_llm_model="exercise-model",
            llm_model="shared-model",
        ),
    )

    payload = asyncio.run(
        quiz_tutor.generate_quiz_tutor_response(
            question="2 + 2 bằng bao nhiêu?",
            options=["3", "4", "5", "6"],
            user_question="Giải thích giúp mình",
        )
    )

    assert payload["explanation"] == "Đây là phần giải thích đủ dài cho học sinh hiểu bài."
    assert payload["structured"] is None
    assert payload["turn_count"] == 1
    messages = captured["input_messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == tutor_chat.TUTOR_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], str)
    assert "Không chào lại" in messages[1]["content"]
    assert captured["max_tokens"] == 1024


def test_quiz_tutor_rejects_image_inputs_when_model_is_text_only(monkeypatch):
    monkeypatch.setattr(tutor_chat, "_resolve_tutor_model", lambda: "deepseek-v4-pro")
    monkeypatch.setattr(tutor_chat, "_tutor_model_supports_vision", lambda _model: False)

    with pytest.raises(ValueError, match="require a vision-capable LLM model"):
        asyncio.run(
            quiz_tutor.generate_quiz_tutor_response(
                question="Câu hỏi dựa vào hình?",
                options=["A", "B", "C", "D"],
                user_question="Giải thích giúp mình",
                question_image="https://example.test/question.png",
            )
        )


def test_quiz_tutor_keeps_image_blocks_when_model_supports_vision(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_generate_tutor_text(*, input_messages, model, timeout_sec, action, max_tokens):
        del model, timeout_sec, action, max_tokens
        captured["input_messages"] = input_messages
        return "Đây là phần giải thích dựa trên hình."

    monkeypatch.setattr(tutor_chat, "generate_tutor_text", fake_generate_tutor_text)
    monkeypatch.setattr(tutor_chat, "_resolve_tutor_model", lambda: "gpt-4o-mini")
    monkeypatch.setattr(tutor_chat, "_tutor_model_supports_vision", lambda _model: True)
    monkeypatch.setattr(
        tutor_chat,
        "get_settings",
        lambda: SimpleNamespace(llm_timeout_sec=5, llm_custom_provider=None),
    )

    asyncio.run(
        quiz_tutor.generate_quiz_tutor_response(
            question="Câu hỏi dựa vào hình?",
            options=["A", "B", "C", "D"],
            user_question="Giải thích giúp mình",
            question_image="https://example.test/question.png",
        )
    )

    messages = captured["input_messages"]
    assert isinstance(messages, list)
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "text"
    assert user_content[1] == {"type": "image", "url": "https://example.test/question.png"}

    normalized = normalize_chat_messages(messages, model="gpt-4o-mini")
    assert normalized[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/question.png"},
    }
