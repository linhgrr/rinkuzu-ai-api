"""
llm.py — Shared LLM helpers for the API codebase.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from api.config import get_settings

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]


def _normalize_openai_base_url(url: str) -> str:
    raw = (url or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def resolve_llm_api_key() -> str | None:
    settings = get_settings()
    return settings.openai_api_key or settings.gemini_api_key or settings.google_api_key


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


def get_llm(temperature: float = 0.0, **kwargs: Any) -> ChatOpenAI:
    """Create a configured LangChain chat model for the current settings."""
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None) or settings.openai_base_url or "http://localhost:6969"
    base_url = _normalize_openai_base_url(base_url_raw)
    model = kwargs.pop("model", None) or settings.openai_model or "gemini-3.0-pro"
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


def get_structured_llm(
    schema: type[StructuredModelT],
    *,
    temperature: float = 0.0,
    method: StructuredOutputMethod = "json_schema",
    strict: bool | None = True,
    **kwargs: Any,
) -> Any:
    """Create a LangChain structured-output runnable using provider-native JSON schema mode."""
    llm = get_llm(temperature=temperature, **kwargs)
    return llm.with_structured_output(schema, method=method, strict=strict)


def serialize_responses_sse_event(event: Any) -> bytes:
    if hasattr(event, "model_dump_json"):
        payload = event.model_dump_json(exclude_none=True)
    elif hasattr(event, "to_json"):
        payload = event.to_json()
    else:
        payload = json.dumps(event, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


def _resolve_shared_llm_model(explicit_model: str | None) -> str | None:
    settings = get_settings()
    return settings.exercise_llm_model or explicit_model or settings.openai_model


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
