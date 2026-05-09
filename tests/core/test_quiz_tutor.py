from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.core.quiz import quiz_tutor


class _FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[list[object]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.response_text)


def test_generate_quiz_tutor_response_uses_langchain_messages(monkeypatch):
    llm = _FakeLLM("Đây là phần giải thích đủ dài cho học sinh hiểu bài.")

    monkeypatch.setattr(
        quiz_tutor,
        "get_llm",
        lambda **_kwargs: llm,
    )
    monkeypatch.setattr(quiz_tutor, "resolve_retry_policy", lambda: (1, 0.0))
    monkeypatch.setattr(
        quiz_tutor,
        "get_settings",
        lambda: SimpleNamespace(
            llm_timeout_sec=5,
            exercise_llm_model="gpt-4o-mini",
            openai_model="gpt-4o-mini",
        ),
    )

    payload = quiz_tutor.generate_quiz_tutor_response(
        question="2 + 2 bằng bao nhiêu?",
        options=["3", "4", "5", "6"],
        user_question="Giải thích giúp mình",
    )

    assert payload["success"] is True
    assert payload["data"]["explanation"] == llm.response_text
    assert isinstance(llm.calls[0][0], SystemMessage)
    assert llm.calls[0][0].content == quiz_tutor.TUTOR_SYSTEM_PROMPT
    assert isinstance(llm.calls[0][1], HumanMessage)
    assert llm.calls[0][1].content[0]["type"] == "text"
