import asyncio
from datetime import UTC, datetime
import time
from types import SimpleNamespace

import pytest

from api.core.shared.document_text import ExtractedDocumentText
from api.domains.quiz import draft_service as draft_service_module
from api.domains.quiz.draft_service import (
    QuizDraftService,
    QuizDraftValidationError,
    public_draft,
)


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
    assert draft["progress"] == {"processed": 0, "total": 3, "percent": 0}


@pytest.mark.asyncio
async def test_create_draft_enqueues_without_reading_object_storage(monkeypatch):
    service = QuizDraftService()
    persisted: list[dict[str, object]] = []
    settings = SimpleNamespace(
        object_storage_bucket="bucket",
        llm_model="quiz-model",
        quiz_extract_max_pdf_bytes=10 * 1024 * 1024,
    )
    req = SimpleNamespace(
        title="Quiz",
        s3_key="uploads/quiz_extract/user-1/file.pdf",
        file_name="file.pdf",
        file_size=1024,
        category_id="category-1",
        description=None,
        prompt=None,
    )

    def fake_get_s3_client():
        return object()

    monkeypatch.setattr(draft_service_module.mongo_store, "is_available", lambda: True)
    monkeypatch.setattr(draft_service_module, "get_settings", lambda: settings)
    monkeypatch.setattr(draft_service_module, "get_s3_client", fake_get_s3_client)
    monkeypatch.setattr(
        draft_service_module,
        "validate_quiz_extract_dependencies",
        lambda _settings, _client: None,
    )

    async def fake_create(doc: dict[str, object]):
        persisted.append(doc)
        return doc

    monkeypatch.setattr(draft_service_module, "create_quiz_draft", fake_create)
    monkeypatch.setattr(
        service,
        "_read_pdf_bytes",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must run in background only")),
    )

    created = await service.create_draft(req, "user-1")

    assert created["status"] == "queued"
    assert created["pdf"] == {
        "s3_key": "uploads/quiz_extract/user-1/file.pdf",
        "file_name": "file.pdf",
        "file_size": 1024,
        "page_count": None,
    }
    assert persisted == [created]


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
async def test_mark_submitted_is_idempotent_for_already_submitted_draft(monkeypatch):
    service = QuizDraftService()
    already = {
        "draft_id": "draft-1",
        "status": "submitted",
        "submitted_quiz_id": "quiz-original",
        "pdf": {"s3_key": "uploads/quiz_extract/user-1/file.pdf"},
    }

    async def fake_get_draft(draft_id: str, user_id: str):
        return already

    def _fail_update(*_args, **_kwargs):
        raise AssertionError("update must not run for an already-submitted draft")

    def _fail_delete(*_args, **_kwargs):
        raise AssertionError("PDF must not be re-deleted for an already-submitted draft")

    monkeypatch.setattr(service, "get_draft", fake_get_draft)
    monkeypatch.setattr(draft_service_module, "update_quiz_draft_for_user", _fail_update)
    monkeypatch.setattr(service, "_delete_pdf_best_effort", _fail_delete)

    result = await service.mark_submitted("draft-1", "user-1", "quiz-retry")

    assert result["submitted_quiz_id"] == "quiz-original"


@pytest.mark.asyncio
async def test_mark_submitted_persists_quiz_id_on_first_submit(monkeypatch):
    service = QuizDraftService()
    draft = {
        "draft_id": "draft-1",
        "status": "completed",
        "submitted_quiz_id": None,
        "pdf": {"s3_key": "uploads/quiz_extract/user-1/file.pdf"},
    }
    updates: list[dict[str, object]] = []
    deleted_keys: list[str | None] = []

    async def fake_get_draft(draft_id: str, user_id: str):
        return draft

    async def fake_update(_draft_id: str, _user_id: str, payload: dict[str, object]):
        updates.append(payload)
        return {**draft, **payload}

    def fake_delete(key: str | None):
        deleted_keys.append(key)

    monkeypatch.setattr(service, "get_draft", fake_get_draft)
    monkeypatch.setattr(draft_service_module, "update_quiz_draft_for_user", fake_update)
    monkeypatch.setattr(service, "_delete_pdf_best_effort", fake_delete)

    result = await service.mark_submitted("draft-1", "user-1", "quiz-new")

    assert updates[0]["status"] == "submitted"
    assert updates[0]["submitted_quiz_id"] == "quiz-new"
    assert result["submitted_quiz_id"] == "quiz-new"
    assert deleted_keys == ["uploads/quiz_extract/user-1/file.pdf"]


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

    async def fake_update_quiz_draft_for_user(
        _draft_id: str, _user_id: str, payload: dict[str, object]
    ):
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

    def _get_s3_client(*, endpoint_url=None):
        del endpoint_url
        return fake_s3_client

    monkeypatch.setattr(draft_service_module, "get_quiz_draft_s3_client", _get_s3_client)
    monkeypatch.setattr(service, "_read_pdf_bytes", lambda *_args: b"%PDF-demo")
    monkeypatch.setattr(service, "_count_pdf_pages", lambda _bytes: 1)

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
    assert updates[1]["pdf"]["file_size"] == len(b"%PDF-demo")
    assert updates[1]["pdf"]["page_count"] == 1
    assert updates[1]["progress"] == {"processed": 1, "total": 3, "percent": 33}
    assert updates[2]["progress"] == {"processed": 2, "total": 3, "percent": 67}
    assert updates[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_process_draft_falls_back_when_primary_source_endpoint_times_out(monkeypatch):
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
    settings = SimpleNamespace(
        object_storage_bucket="bucket",
        llm_model="quiz-model",
        object_storage_client_endpoint="http://internal-storage",
        object_storage_public_base_url="https://public-storage",
        object_storage_endpoint_internal="internal-storage",
        quiz_extract_source_download_timeout_sec=1,
        quiz_extract_source_endpoint_timeout_sec=0.001,
    )

    async def fake_get_draft(_draft_id: str, _user_id: str):
        return draft

    async def fake_update_quiz_draft_for_user(
        _draft_id: str, _user_id: str, payload: dict[str, object]
    ):
        updates.append(payload)
        return {"status": payload.get("status")}

    def fake_get_quiz_draft_s3_client(*, endpoint_url: str | None = None):
        return SimpleNamespace(endpoint_url=endpoint_url)

    def fake_read_pdf_bytes(s3_client, *_args):
        if s3_client.endpoint_url == "http://internal-storage":
            time.sleep(0.05)
            return b"%PDF-internal"
        return b"%PDF-public"

    async def fake_load_or_extract_document_text(**_kwargs):
        return extracted

    async def fake_invoke_document_text_extract_llm(**_kwargs):
        return [
            {
                "question": "Q1",
                "type": "single",
                "options": ["A", "B", "C", "D"],
                "correctIndex": 0,
            }
        ]

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
    monkeypatch.setattr(draft_service_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        draft_service_module,
        "get_quiz_draft_s3_client",
        fake_get_quiz_draft_s3_client,
    )
    monkeypatch.setattr(service, "_read_pdf_bytes", fake_read_pdf_bytes)
    monkeypatch.setattr(service, "_count_pdf_pages", lambda _bytes: 1)
    monkeypatch.setattr(
        service,
        "_load_or_extract_document_text",
        fake_load_or_extract_document_text,
    )
    monkeypatch.setattr(
        draft_service_module,
        "invoke_document_text_extract_llm",
        fake_invoke_document_text_extract_llm,
    )

    await service.process_draft("draft-1", "user-1")

    assert updates[1]["pdf"]["file_size"] == len(b"%PDF-public")
    assert updates[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_process_draft_marks_source_timeout_as_failed(monkeypatch):
    service = QuizDraftService()
    failures: list[str] = []

    async def fake_run(_draft_id: str, _user_id: str):
        await asyncio.sleep(1)

    async def fake_mark_failed(_draft_id: str, _user_id: str, message: str):
        failures.append(message)

    monkeypatch.setattr(service, "_run_processing_stages", fake_run)
    monkeypatch.setattr(service, "_mark_failed", fake_mark_failed)
    monkeypatch.setattr(draft_service_module, "QUIZ_DRAFT_JOB_TIMEOUT_SEC", 0.001)

    await service.process_draft("draft-1", "user-1")

    assert failures == ["Quiz extraction exceeded 0.001 seconds."]


@pytest.mark.asyncio
async def test_process_draft_requeues_cancelled_task(monkeypatch):
    service = QuizDraftService()
    interrupted: list[tuple[str, str]] = []

    async def fake_run(_draft_id: str, _user_id: str):
        raise asyncio.CancelledError

    async def fake_mark_interrupted(draft_id: str, user_id: str):
        interrupted.append((draft_id, user_id))

    monkeypatch.setattr(service, "_run_processing_stages", fake_run)
    monkeypatch.setattr(service, "_mark_interrupted", fake_mark_interrupted)

    with pytest.raises(asyncio.CancelledError):
        await service.process_draft("draft-1", "user-1")

    assert interrupted == [("draft-1", "user-1")]


@pytest.mark.asyncio
async def test_interruption_does_not_requeue_terminal_draft(monkeypatch):
    service = QuizDraftService()

    async def fake_load(_draft_id: str, _user_id: str):
        return {"status": "completed"}

    async def fail_update(*_args, **_kwargs):
        raise AssertionError("terminal draft must not be requeued")

    monkeypatch.setattr(draft_service_module, "load_quiz_draft_for_user", fake_load)
    monkeypatch.setattr(draft_service_module, "update_quiz_draft_for_user", fail_update)

    await service._mark_interrupted("draft-1", "user-1")
