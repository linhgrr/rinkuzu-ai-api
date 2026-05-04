"""
llm.py — Shared LLM helpers for the API codebase.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from api.config import get_settings

_llm_state: dict[str, Any] = {
    "shared_llm": None,
    "structured_llms": {},
}


def _normalize_openai_base_url(url: str) -> str:
    raw = (url or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def resolve_llm_api_key() -> str:
    settings = get_settings()
    return settings.llm_api_key or settings.gemini_api_key or settings.google_api_key


def resolve_retry_policy() -> tuple[int, float]:
    settings = get_settings()
    return (
        max(1, int(settings.llm_retry_attempts)),
        max(0.0, float(settings.llm_retry_backoff_sec)),
    )


def sleep_before_retry(attempt: int, base_delay_sec: float) -> None:
    if base_delay_sec <= 0:
        return
    time.sleep(base_delay_sec * attempt)


def _ngrok_headers() -> dict[str, str]:
    return {"ngrok-skip-browser-warning": "true"}


def get_llm(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Create a configured ChatOpenAI client for the current settings."""
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None) or settings.llm_base_url or "http://localhost:6969"
    base_url = _normalize_openai_base_url(base_url_raw)
    model = kwargs.pop("model", None) or settings.llm_model or "gemini-3.0-pro"
    api_key = kwargs.pop("api_key", None) or resolve_llm_api_key()
    timeout = kwargs.pop("timeout", settings.llm_timeout_sec)
    max_retries = kwargs.pop("max_retries", settings.llm_max_retries)

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        default_headers=_ngrok_headers(),
        **kwargs,
    )


def get_embeddings(**kwargs) -> OpenAIEmbeddings:
    """Create a configured OpenAIEmbeddings client for the current settings."""
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None) or settings.llm_base_url or "http://localhost:6969"
    base_url = _normalize_openai_base_url(base_url_raw)
    model = kwargs.pop("model", None) or settings.llm_embedding_model
    api_key = kwargs.pop("api_key", None) or resolve_llm_api_key()
    timeout = kwargs.pop("timeout", settings.llm_timeout_sec)

    return OpenAIEmbeddings(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        default_headers=_ngrok_headers(),
        **kwargs,
    )


def build_chat_completions_url(base_url: str | None) -> str:
    if not base_url:
        raise RuntimeError("LLM base URL is not configured")
    return f"{_normalize_openai_base_url(base_url)}/chat/completions"


def _resolve_shared_llm_model(explicit_model: str | None) -> str | None:
    settings = get_settings()
    return settings.exercise_llm_model or explicit_model


def initialize_shared_llm(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Initialize and cache the shared adaptive-learning ChatOpenAI client."""
    shared_llm = get_llm(
        temperature=0.3,
        base_url=base_url,
        model=_resolve_shared_llm_model(model),
        api_key=api_key,
    )
    _llm_state["shared_llm"] = shared_llm
    _llm_state["structured_llms"] = {}

    logger.info(f"[LLM] Connecting with model={shared_llm.model_name}")
    logger.info("[LLM] ✓ Shared runtime ready.")
    return shared_llm


def get_shared_llm() -> ChatOpenAI:
    """Return the cached shared client, lazily initializing if needed."""
    if _llm_state["shared_llm"] is None:
        _llm_state["shared_llm"] = initialize_shared_llm()
    return _llm_state["shared_llm"]


def get_structured_llm(schema: Any, *, method: str = "json_mode") -> Any:
    """Return a cached structured-output wrapper for the shared client."""
    llm = get_shared_llm()
    cache_key = (
        f"{getattr(schema, '__module__', '')}:{getattr(schema, '__name__', repr(schema))}:{method}"
    )
    structured_llms = _llm_state["structured_llms"]
    if cache_key not in structured_llms:
        structured_llms[cache_key] = llm.with_structured_output(schema, method=method)
    return structured_llms[cache_key]


def extract_llm_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()
