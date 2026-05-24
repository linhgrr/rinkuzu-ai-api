from types import SimpleNamespace

import pytest

from api.core.shared.document_text import (
    DocumentPageText,
    DocumentTextConfigurationError,
    ExtractedDocumentText,
    LandingAIDocumentTextExtractor,
    OCRApiConfig,
    _landing_ai_to_extracted_text,
    build_document_text_extractor,
    build_ocr_api_config,
    load_or_extract_document_text_cached,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload
        self.status_code = 200
        self.text = "ok"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        data: dict[str, object],
        files: dict[str, tuple[str, bytes, str]],
    ) -> _FakeResponse:
        self.calls.append({"url": url, "headers": headers, "data": data, "files": files})
        return _FakeResponse(self.payload)


def test_build_ocr_api_config_requires_api_key():
    with pytest.raises(DocumentTextConfigurationError, match="OCR_API_KEY"):
        build_ocr_api_config(
            SimpleNamespace(
                ocr_base_url="https://api.va.landing.ai/v1/ade/parse",
                ocr_api_key=None,
                ocr_model="dpt-2-mini",
                ocr_timeout_sec=120,
            )
        )


def test_build_document_text_extractor_returns_landing_ai_extractor():
    extractor = build_document_text_extractor(
        SimpleNamespace(
            ocr_base_url="https://api.va.landing.ai/v1/ade/parse",
            ocr_api_key="secret",
            ocr_model="dpt-2-mini",
            ocr_timeout_sec=90,
        )
    )

    assert isinstance(extractor, LandingAIDocumentTextExtractor)
    assert extractor.config.timeout_sec == 90


def test_landing_ai_payload_is_converted_into_page_text():
    payload = {
        "markdown": "full markdown",
        "splits": [
            {"pages": [1], "markdown": "<a id='a'></a>\n\nXin chào"},
            {"pages": [2], "markdown": "Thế giới"},
        ],
        "metadata": {
            "page_count": 2,
            "credit_usage": 3.0,
            "job_id": "job-1",
            "version": "dpt-2-mini",
            "failed_pages": [],
        },
    }

    extracted = _landing_ai_to_extracted_text(payload, file_name="scan.pdf")

    assert extracted.metadata["source"] == "ocr_api"
    assert extracted.metadata["provider"] == "landingai"
    assert extracted.metadata["page_count"] == 2
    assert extracted.pages[0].text == "Xin chào"
    assert extracted.pages[1].page_number == 2
    assert extracted.metadata["credit_usage"] == 3.0


def test_landing_ai_document_text_extractor_posts_pdf_to_api():
    fake_client = _FakeClient(
        {
            "markdown": "Nội dung OCR",
            "splits": [{"pages": [1], "markdown": "Nội dung OCR"}],
            "metadata": {"page_count": 1, "credit_usage": 1.5, "job_id": "job-2", "version": "dpt-2-mini"},
        }
    )
    extractor = LandingAIDocumentTextExtractor(
        config=OCRApiConfig(
            endpoint="https://api.va.landing.ai/v1/ade/parse",
            api_key="secret",
            model="dpt-2-mini",
            timeout_sec=30,
        ),
        client=fake_client,  # type: ignore[arg-type]
    )

    result = extractor.extract_bytes(b"%PDF-demo", filename="scan.pdf")

    assert result.metadata["page_count"] == 1
    assert "Nội dung OCR" in result.text
    assert len(fake_client.calls) == 1
    request = fake_client.calls[0]
    assert request["url"] == "https://api.va.landing.ai/v1/ade/parse"
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert request["data"] == {"model": "dpt-2-mini", "split": "page"}
    file_payload = request["files"]["document"]
    assert file_payload[0] == "scan.pdf"
    assert file_payload[1] == b"%PDF-demo"
    assert file_payload[2] == "application/pdf"


@pytest.mark.asyncio
async def test_load_or_extract_document_text_cached_returns_cached_record(monkeypatch):
    cached_record = {
        "file_hash": "hash-1",
        "file_name": "scan.pdf",
        "file_size_bytes": 12,
        "text": "## Trang 1\nXin chào",
        "page_count": 1,
        "provider": "landingai",
        "model": "dpt-2-mini",
        "pages": [{"page_number": 1, "text": "Xin chào"}],
        "metadata": {"page_count": 1, "provider": "landingai", "model": "dpt-2-mini"},
    }

    monkeypatch.setattr("api.core.shared.document_text.mongo_store.is_available", lambda: True)

    async def fake_load_document_ocr_record(file_hash: str):
        assert file_hash == "hash-1"
        return cached_record

    monkeypatch.setattr(
        "api.core.shared.document_text.load_document_ocr_record",
        fake_load_document_ocr_record,
    )
    monkeypatch.setattr(
        "api.core.shared.document_text.save_document_ocr_record",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not persist cache hit")),
    )

    document_text = await load_or_extract_document_text_cached(
        file_hash="hash-1",
        file_name="scan.pdf",
        file_size_bytes=12,
        extract_document_text=_fail_async,
    )

    assert document_text.text == "## Trang 1\nXin chào"
    assert document_text.metadata["ocr_cache_hit"] is True
    assert document_text.metadata["file_hash"] == "hash-1"


@pytest.mark.asyncio
async def test_load_or_extract_document_text_cached_persists_cache_miss(monkeypatch):
    extracted = ExtractedDocumentText(
        text="## Trang 1\nNội dung",
        pages=[DocumentPageText(page_number=1, text="Nội dung")],
        metadata={"page_count": 1, "provider": "landingai", "model": "dpt-2-mini"},
    )
    saved: dict[str, object] = {}

    monkeypatch.setattr("api.core.shared.document_text.mongo_store.is_available", lambda: True)

    async def fake_load_document_ocr_record(_file_hash: str):
        return None

    async def fake_save_document_ocr_record(**kwargs):
        saved.update(kwargs)
        return True

    monkeypatch.setattr(
        "api.core.shared.document_text.load_document_ocr_record",
        fake_load_document_ocr_record,
    )
    monkeypatch.setattr(
        "api.core.shared.document_text.save_document_ocr_record",
        fake_save_document_ocr_record,
    )

    resolved_size_calls = 0

    async def fake_resolve_file_size_bytes():
        nonlocal resolved_size_calls
        resolved_size_calls += 1
        return 456

    document_text = await load_or_extract_document_text_cached(
        file_hash="hash-2",
        file_name="scan.pdf",
        extract_document_text=lambda: _async_value(extracted),
        resolve_file_size_bytes=fake_resolve_file_size_bytes,
    )

    assert document_text is extracted
    assert resolved_size_calls == 1
    assert saved["file_hash"] == "hash-2"
    assert saved["file_name"] == "scan.pdf"
    assert saved["file_size_bytes"] == 456


async def _async_value(value):
    return value


async def _fail_async():
    raise AssertionError("should not OCR")
