from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from api.core.quiz import draft_service as draft_service_module
from api.core.quiz.draft_service import (
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)
from api.core.shared.document_text import ExtractedDocumentText


def test_quiz_draft_s3_key_must_belong_to_user():
    with pytest.raises(QuizDraftValidationError):
        QuizDraftService._normalize_and_validate_s3_key(
            "uploads/quiz_extract/user-2/file.pdf",
            "user-1",
        )


def test_public_draft_uses_safe_defaults():
    now = datetime.now(UTC)

    draft = public_draft(
        {
            "draft_id": "draft-1",
            "title": "Quiz",
            "status": "queued",
            "pdf": {"s3_key": "uploads/quiz_extract/user-1/file.pdf"},
            "created_at": now,
            "updated_at": now,
            "expires_at": now,
        }
    )

    assert draft["draft_id"] == "draft-1"
    assert draft["questions"] == []
    assert draft["progress"] == {"processed": 0, "total": 1, "percent": 0}


@pytest.mark.asyncio
async def test_load_or_extract_document_text_reuses_cached_ocr_record(monkeypatch):
    service = QuizDraftService()
    cached_text = ExtractedDocumentText(
        text="## Trang 1\nXin chào",
        pages=[],
        metadata={"page_count": 1, "ocr_cache_hit": True, "file_hash": "hash-1"},
    )

    monkeypatch.setattr(draft_service_module, "calculate_pdf_bytes_hash", lambda _bytes: "hash-1")
    async def fake_load_or_extract_document_text_cached(
        *,
        file_hash: str,
        file_name: str,
        extract_document_text,
        file_size_bytes: int | None = None,
        resolve_file_size_bytes=None,
    ):
        assert file_hash == "hash-1"
        assert file_name == "lesson.pdf"
        assert file_size_bytes == len(b"%PDF-demo")
        del extract_document_text, resolve_file_size_bytes
        return cached_text
    monkeypatch.setattr(
        draft_service_module,
        "load_or_extract_document_text_cached",
        fake_load_or_extract_document_text_cached,
    )

    document_text = await service._load_or_extract_document_text(
        pdf_bytes=b"%PDF-demo",
        filename="lesson.pdf",
    )

    assert document_text is cached_text


@pytest.mark.asyncio
async def test_load_or_extract_document_text_persists_on_cache_miss(monkeypatch):
    service = QuizDraftService()
    extracted = ExtractedDocumentText(
        text="## Trang 1\nNội dung",
        pages=[],
        metadata={"page_count": 1, "provider": "landingai", "model": "dpt-2-mini"},
    )

    monkeypatch.setattr(draft_service_module, "calculate_pdf_bytes_hash", lambda _bytes: "hash-2")
    async def fake_load_or_extract_document_text_cached(
        *,
        file_hash: str,
        file_name: str,
        extract_document_text,
        file_size_bytes: int | None = None,
        resolve_file_size_bytes=None,
    ):
        del resolve_file_size_bytes
        assert file_hash == "hash-2"
        assert file_name == "lesson.pdf"
        assert file_size_bytes == len(b"%PDF-demo")
        return await extract_document_text()

    monkeypatch.setattr(
        draft_service_module,
        "load_or_extract_document_text_cached",
        fake_load_or_extract_document_text_cached,
    )
    monkeypatch.setattr(
        draft_service_module,
        "extract_document_text_from_bytes",
        lambda pdf_bytes, *, filename=None: (
            extracted
            if pdf_bytes == b"%PDF-demo" and filename == "lesson.pdf"
            else (_ for _ in ()).throw(AssertionError("unexpected OCR input"))
        ),
    )

    document_text = await service._load_or_extract_document_text(
        pdf_bytes=b"%PDF-demo",
        filename="lesson.pdf",
    )

    assert document_text is extracted


@pytest.mark.asyncio
async def test_process_draft_uses_document_text_flow(monkeypatch):
    service = QuizDraftService()
    updates: list[dict[str, object]] = []
    extracted = ExtractedDocumentText(
        text="## Trang 1\nNội dung",
        pages=[],
        metadata={"page_count": 1},
    )

    draft = {
        "draft_id": "draft-1",
        "status": "queued",
        "prompt": None,
        "pdf": {
            "s3_key": "uploads/quiz_extract/user-1/file.pdf",
            "file_name": "file.pdf",
        },
    }

    async def fake_get_draft(draft_id: str, user_id: str):
        assert draft_id == "draft-1"
        assert user_id == "user-1"
        return draft

    async def fake_update_quiz_draft_for_user(_draft_id: str, _user_id: str, payload: dict[str, object]):
        updates.append(payload)
        return {"status": payload.get("status")}

    monkeypatch.setattr(service, "get_draft", fake_get_draft)
    monkeypatch.setattr(
        draft_service_module,
        "update_quiz_draft_for_user",
        fake_update_quiz_draft_for_user,
    )
    monkeypatch.setattr(
        draft_service_module,
        "load_quiz_draft_for_user",
        lambda *_args, **_kwargs: fake_get_draft("draft-1", "user-1"),
    )
    monkeypatch.setattr(
        draft_service_module,
        "get_settings",
        lambda: SimpleNamespace(object_storage_bucket="bucket", llm_model="quiz-model"),
    )
    fake_s3_client = object()
    def _get_s3_client():
        return fake_s3_client

    monkeypatch.setattr(draft_service_module, "get_s3_client", _get_s3_client)
    monkeypatch.setattr(service, "_read_pdf_bytes", lambda *_args: b"%PDF-demo")

    async def fake_load_or_extract_document_text(**_kwargs):
        return extracted

    monkeypatch.setattr(
        service,
        "_load_or_extract_document_text",
        fake_load_or_extract_document_text,
    )

    async def fake_invoke_document_text_extract_llm(**kwargs):
        assert kwargs["document_text"] is extracted
        assert kwargs["filename"] == "file.pdf"
        return [
            {
                "question": "Q1",
                "type": "single",
                "options": ["A", "B", "C", "D"],
                "correctIndex": 0,
            }
        ]

    monkeypatch.setattr(
        draft_service_module,
        "invoke_document_text_extract_llm",
        fake_invoke_document_text_extract_llm,
    )

    await service.process_draft("draft-1", "user-1")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "completed"
