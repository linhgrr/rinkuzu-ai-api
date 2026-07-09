import asyncio
from types import SimpleNamespace

from api.domains.quiz import quiz_tutor


def test_generate_quiz_tutor_response_uses_project_standard_message_shape(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_generate_tutor_text(*, input_messages, model, timeout_sec, action):
        captured["model"] = model
        captured["input_messages"] = input_messages
        captured["timeout_sec"] = timeout_sec
        captured["action"] = action
        return "Đây là phần giải thích đủ dài cho học sinh hiểu bài."

    monkeypatch.setattr(quiz_tutor, "generate_tutor_text", fake_generate_tutor_text)
    monkeypatch.setattr(
        quiz_tutor,
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
    assert messages[0]["content"] == quiz_tutor.TUTOR_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "text"
