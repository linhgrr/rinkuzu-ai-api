"""LLM adapter for the content pipeline.

This module intentionally keeps a thin wrapper around the shared backend LLM
helpers so content-pipeline tests can monkeypatch the adapter surface directly.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from api.config import get_settings


def _normalize_openai_base_url(url: str) -> str:
    raw = (url or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def _resolve_llm_api_key() -> str | None:
    settings = get_settings()
    return settings.llm_api_key or settings.gemini_api_key or settings.google_api_key


def _ngrok_headers() -> dict[str, str]:
    return {"ngrok-skip-browser-warning": "true"}


def get_llm(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Create a ChatOpenAI client using unified backend settings."""
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None) or settings.llm_base_url or "http://localhost:6969"
    model = kwargs.pop("model", None) or settings.llm_model or "gemini-3.0-pro"
    api_key = kwargs.pop("api_key", None) or _resolve_llm_api_key()
    timeout = kwargs.pop("timeout", settings.llm_timeout_sec)
    max_retries = kwargs.pop("max_retries", settings.llm_max_retries)

    return ChatOpenAI(
        base_url=_normalize_openai_base_url(base_url_raw),
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=_ngrok_headers(),
        **kwargs,
    )


def get_embeddings(**kwargs) -> OpenAIEmbeddings:
    """Create an embeddings client using unified backend settings."""
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None) or settings.llm_base_url or "http://localhost:6969"
    model = kwargs.pop("model", None) or settings.llm_embedding_model
    api_key = kwargs.pop("api_key", None) or _resolve_llm_api_key()
    timeout = kwargs.pop("timeout", settings.llm_timeout_sec)

    return OpenAIEmbeddings(
        base_url=_normalize_openai_base_url(base_url_raw),
        model=model,
        api_key=api_key,
        timeout=timeout,
        default_headers=_ngrok_headers(),
        **kwargs,
    )


__all__ = ["ChatOpenAI", "OpenAIEmbeddings", "get_embeddings", "get_llm", "get_settings"]
