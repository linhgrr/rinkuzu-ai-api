import os
from types import SimpleNamespace

from api.core.content_pipeline.infrastructure import llm as llm_module
from api.core.content_pipeline.infrastructure.processors.loaders import (
    pdf_loader,
    vision_pdf_loader,
)


def test_get_llm_uses_unified_backend_settings(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="http://llm.internal",
            llm_model="gemini-test",
            llm_api_key="llm-key",
            gemini_api_key=None,
            google_api_key=None,
            llm_timeout_sec=42,
            llm_max_retries=7,
            llm_embedding_model="embedding-test",
        ),
    )

    class _ChatOpenAIStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", _ChatOpenAIStub)

    llm_module.get_llm(temperature=0.25)

    assert captured["base_url"] == "http://llm.internal/v1"
    assert captured["model"] == "gemini-test"
    assert captured["api_key"] == "llm-key"
    assert captured["temperature"] == 0.25
    assert captured["timeout"] == 42
    assert captured["max_retries"] == 7


def test_get_llm_allows_timeout_and_retry_overrides(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="http://llm.internal",
            llm_model="gemini-test",
            llm_api_key="llm-key",
            gemini_api_key=None,
            google_api_key=None,
            llm_timeout_sec=42,
            llm_max_retries=7,
            llm_embedding_model="embedding-test",
        ),
    )

    class _ChatOpenAIStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", _ChatOpenAIStub)

    llm_module.get_llm(timeout=150, max_retries=1)

    assert captured["timeout"] == 150
    assert captured["max_retries"] == 1


def test_get_embeddings_uses_unified_backend_settings(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="http://llm.internal",
            llm_model="unused",
            llm_api_key=None,
            gemini_api_key="gemini-key",
            google_api_key=None,
            llm_timeout_sec=24,
            llm_max_retries=2,
            llm_embedding_model="text-embedding-test",
        ),
    )

    class _OpenAIEmbeddingsStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "OpenAIEmbeddings", _OpenAIEmbeddingsStub)

    llm_module.get_embeddings()

    assert captured["base_url"] == "http://llm.internal/v1"
    assert captured["model"] == "text-embedding-test"
    assert captured["api_key"] == "gemini-key"
    assert captured["timeout"] == 24


def test_get_embeddings_allows_timeout_override(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="http://llm.internal",
            llm_model="unused",
            llm_api_key="llm-key",
            gemini_api_key=None,
            google_api_key=None,
            llm_timeout_sec=24,
            llm_max_retries=2,
            llm_embedding_model="text-embedding-test",
        ),
    )

    class _OpenAIEmbeddingsStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "OpenAIEmbeddings", _OpenAIEmbeddingsStub)

    llm_module.get_embeddings(timeout=99)

    assert captured["timeout"] == 99


def test_pdf_loader_prefers_backend_settings_for_api_key(monkeypatch):
    monkeypatch.setattr(pdf_loader, "AGENTIC_DOC_AVAILABLE", True)
    monkeypatch.setattr(
        pdf_loader,
        "get_settings",
        lambda: SimpleNamespace(vision_agent_api_key="vision-key"),
    )

    loader = pdf_loader.PDFLoader()

    assert loader.api_key == "vision-key"
    assert os.environ["VISION_AGENT_API_KEY"] == "vision-key"


def test_vision_pdf_loader_uses_unified_backend_settings(monkeypatch):
    sentinel_client = object()
    monkeypatch.setattr(vision_pdf_loader, "_get_s3_client", lambda: sentinel_client)
    monkeypatch.setattr(
        vision_pdf_loader,
        "get_settings",
        lambda: SimpleNamespace(
            pdf_ocr_concurrency=8,
            llm_base_url="http://llm.internal",
            llm_model="ocr-model",
            llm_api_key=None,
            gemini_api_key=None,
            google_api_key="google-key",
            vision_pdf_request_timeout_sec=33,
            llm_max_retries=5,
            s3_bucket_name="lesson-bucket",
            s3_endpoint_url="http://minio.internal",
            s3_access_key_id="access",
            s3_secret_access_key="secret",
        ),
    )

    loader = vision_pdf_loader.VisionPDFLoader()

    assert loader.concurrency == 8
    assert loader.base_url == "http://llm.internal"
    assert loader.model == "ocr-model"
    assert loader.api_key == "google-key"
    assert loader.request_timeout_sec == 33
    assert loader.max_retries == 5
    assert loader.bucket_name == "lesson-bucket"
    assert loader.s3_endpoint_url == "http://minio.internal"
    assert loader.s3_client is sentinel_client
