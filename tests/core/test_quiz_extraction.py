from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from api.core.shared.document_text import DocumentPageText, ExtractedDocumentText
from api.domains.quiz import extraction
from api.domains.quiz.extraction import (
    ExtractedQuizQuestion,
    ExtractedQuizQuestionBatch,
)


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
    monkeypatch.setattr(
        extraction, "invoke_structured_completion", fake_invoke_structured_completion
    )
    monkeypatch.setattr(
        extraction,
        "get_settings",
        lambda: SimpleNamespace(quiz_extract_max_chars=200_000),
    )

    result = extraction._extract_questions_from_pdf_bytes_sync(
        b"%PDF-demo",
        "sample.pdf",
        "extract quiz",
        "quiz-model",
        5.0,
    )

    assert result[0]["correctIndex"] == 2
    assert len(captured) == 1
    first_request = captured[0]
    assert first_request["schema"] is ExtractedQuizQuestionBatch
    messages = first_request["messages"]
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert "<document_text>" in content
    assert "Tên file: sample.pdf" in content
    # Single call carries the full OCR text, both pages in one request.
    assert "Câu 1" in content
    assert "Câu 2" in content


def test_extract_questions_clamps_oversized_document_text(monkeypatch):
    captured: list[dict[str, object]] = []

    def fake_invoke_structured_completion(**kwargs):
        captured.append(kwargs)
        return ExtractedQuizQuestionBatch(questions=[])

    long_text = "x" * 500
    monkeypatch.setattr(
        extraction, "invoke_structured_completion", fake_invoke_structured_completion
    )
    monkeypatch.setattr(
        extraction,
        "get_settings",
        lambda: SimpleNamespace(quiz_extract_max_chars=100),
    )

    extraction._extract_questions_from_document_text_sync(
        ExtractedDocumentText(
            text=long_text,
            pages=[DocumentPageText(page_number=1, text=long_text)],
            metadata={"file_name": "big.pdf", "page_count": 1},
        ),
        "big.pdf",
        "extract quiz",
        "quiz-model",
        5.0,
    )

    assert len(captured) == 1
    content = captured[0]["messages"][1]["content"]
    assert "x" * 100 in content
    assert "x" * 101 not in content
