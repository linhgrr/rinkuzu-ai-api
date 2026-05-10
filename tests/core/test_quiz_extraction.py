from pydantic import ValidationError
import pytest

from api.core.quiz import extraction
from api.core.quiz.extraction import ExtractedQuizQuestion, ExtractedQuizQuestionList


def test_extracted_quiz_question_requires_single_correct_index():
    question = ExtractedQuizQuestion(
        question="Thủ đô của Pháp là gì?",
        type="single",
        options=["London", "Berlin", "Paris", "Madrid"],
        correctIndex=2,
    )

    assert question.to_public_dict()["correctIndex"] == 2


def test_extracted_quiz_question_rejects_invalid_multiple_indexes():
    with pytest.raises(ValidationError):
        ExtractedQuizQuestion(
            question="Những ngôn ngữ lập trình nào sau đây là đúng?",
            type="multiple",
            options=["JavaScript", "HTML", "Python", "CSS"],
            correctIndexes=[0, 4],
        )


def test_pdf_quiz_extraction_uses_langchain_structured_output(monkeypatch):
    captured: dict[str, object] = {}

    class FakeStructuredLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return ExtractedQuizQuestionList(
                [
                    ExtractedQuizQuestion(
                        question="Thủ đô của Pháp là gì?",
                        type="single",
                        options=["London", "Berlin", "Paris", "Madrid"],
                        correctIndex=2,
                    )
                ]
            )

    def _fake_get_structured_llm(*_args, **_kwargs):
        return FakeStructuredLLM()

    monkeypatch.setattr(
        extraction,
        "get_structured_llm",
        _fake_get_structured_llm,
    )

    result = extraction._invoke_pdf_extract_llm_sync(
        b"%PDF-demo",
        "sample.pdf",
        "extract quiz",
        "gpt-4o-mini",
        5.0,
    )

    assert result[0]["correctIndex"] == 2
    assert captured["messages"][0].content[1]["type"] == "file"
