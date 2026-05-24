from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from api.core.quiz import extraction
from api.core.quiz.extraction import (
    ExtractedQuizQuestion,
    ExtractedQuizQuestionBatch,
)
from api.core.shared.document_text import DocumentPageText, ExtractedDocumentText


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


def test_extract_questions_from_pdf_bytes_uses_extracted_document_text(monkeypatch):
    captured: list[dict[str, object]] = []

    def fake_invoke_structured_completion(**kwargs):
        captured.append(kwargs)
        return ExtractedQuizQuestionBatch(
            questions=[
                ExtractedQuizQuestion(
                    question="Thủ đô của Pháp là gì?",
                    type="single",
                    options=["London", "Berlin", "Paris", "Madrid"],
                    correctIndex=2,
                )
            ]
        )

    monkeypatch.setattr(
        extraction,
        "extract_document_text_from_bytes",
        lambda _pdf_bytes, *, filename=None: ExtractedDocumentText(
            text="## Trang 1\nCâu 1\n\n## Trang 2\nCâu 2",
            pages=[
                DocumentPageText(page_number=1, text="Câu 1"),
                DocumentPageText(page_number=2, text="Câu 2"),
            ],
            metadata={"file_name": filename or "sample.pdf", "page_count": 2},
        ),
    )
    monkeypatch.setattr(extraction, "invoke_structured_completion", fake_invoke_structured_completion)
    monkeypatch.setattr(
        extraction,
        "get_settings",
        lambda: SimpleNamespace(content_pipeline_pdf_page_batch_size=1),
    )

    result = extraction._extract_questions_from_pdf_bytes_sync(
        b"%PDF-demo",
        "sample.pdf",
        "extract quiz",
        "quiz-model",
        5.0,
    )

    assert result[0]["correctIndex"] == 2
    assert len(captured) == 2
    first_request = captured[0]
    assert first_request["schema"] is ExtractedQuizQuestionBatch
    messages = first_request["messages"]
    assert messages[1]["role"] == "user"
    assert "<document_text>" in messages[1]["content"]
    assert "Tên file: sample.pdf" in messages[1]["content"]
