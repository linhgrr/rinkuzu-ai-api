"""
llm.py — Shared LLM helpers for the API codebase.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable  # noqa: TC003
import json
import time
from typing import Any, Literal, TypeVar

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from api.config import get_settings

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]
_T = TypeVar("_T")


def _normalize_openai_base_url(url: str) -> str:
    raw = (url or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


class LLMConfigurationError(ValueError):
    """Raised when required LLM settings are missing."""


def resolve_llm_api_key() -> str:
    settings = get_settings()
    key = settings.openai_api_key
    if not key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set. Configure it via environment or .env file."
        )
    return key


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


def with_llm_retry(
    *,
    label: str,
    fn: Callable[[], _T],
    on_exhausted: Callable[[], _T] | None = None,
) -> _T:
    """Invoke *fn* with retries, raising RuntimeError when all attempts fail.

    Pass *on_exhausted* to return a fallback instead of raising.
    """
    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("[LLM] ⏳ {} attempt {}/{}", label, attempt, max_retries)
            result = fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[LLM] ⚠ {} attempt {}/{} failed (will_retry={}): {}",
                label, attempt, max_retries, attempt < max_retries, exc,
            )
            if attempt < max_retries:
                sleep_before_retry(attempt, backoff_sec)
            continue
        elapsed = time.time() - t0
        logger.info("[LLM] ✓ {} in {:.2f}s", label, elapsed)
        return result

    elapsed = time.time() - t0
    logger.error("[LLM] ✗ {} failed after {:.2f}s", label, elapsed)
    if on_exhausted is not None:
        return on_exhausted()
    raise RuntimeError(f"{label} is temporarily unavailable") from last_exc


async def awith_llm_retry(
    *,
    label: str,
    fn: Callable[[], Any],
    on_exhausted: Callable[[], Any] | None = None,
) -> Any:
    """Async variant of :func:`with_llm_retry`."""
    t0 = time.time()
    max_retries, backoff_sec = resolve_retry_policy()
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("[LLM] ⏳ {} attempt {}/{}", label, attempt, max_retries)
            result = await fn() if asyncio.iscoroutinefunction(fn) else fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[LLM] ⚠ {} attempt {}/{} failed (will_retry={}): {}",
                label, attempt, max_retries, attempt < max_retries, exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff_sec * attempt)
            continue
        elapsed = time.time() - t0
        logger.info("[LLM] ✓ {} in {:.2f}s", label, elapsed)
        return result

    elapsed = time.time() - t0
    logger.error("[LLM] ✗ {} failed after {:.2f}s", label, elapsed)
    if on_exhausted is not None:
        return await on_exhausted() if asyncio.iscoroutinefunction(on_exhausted) else on_exhausted()
    raise RuntimeError(f"{label} is temporarily unavailable") from last_exc


def _ngrok_headers() -> dict[str, str]:
    return {"ngrok-skip-browser-warning": "true"}


def get_llm(temperature: float = 0.0, **kwargs: Any) -> ChatOpenAI:
    """Create a configured LangChain chat model for the current settings.

    Raises LLMConfigurationError if required settings (api_key, model) are missing.
    """
    settings = get_settings()

    base_url_raw = kwargs.pop("base_url", None) or settings.openai_base_url
    base_url = _normalize_openai_base_url(base_url_raw) if base_url_raw else None

    model = kwargs.pop("model", None) or settings.openai_model
    if not model:
        raise LLMConfigurationError(
            "OPENAI_MODEL is not set. Configure it via environment or .env file."
        )

    api_key = kwargs.pop("api_key", None) or resolve_llm_api_key()
    timeout = kwargs.pop("timeout", settings.llm_timeout_sec)
    max_retries = kwargs.pop("max_retries", settings.llm_max_retries)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        **({"base_url": base_url} if base_url else {}),
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


def serialize_responses_sse_event(event: object) -> bytes:
    if hasattr(event, "model_dump_json"):
        payload = event.model_dump_json(exclude_none=True)
    elif hasattr(event, "to_json"):
        payload = event.to_json()
    else:
        payload = json.dumps(event, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


def _resolve_shared_llm_model(explicit_model: str | None) -> str:
    settings = get_settings()
    model = settings.exercise_llm_model or explicit_model or settings.openai_model
    if not model:
        raise LLMConfigurationError(
            "OPENAI_MODEL is not set. Configure it via environment or .env file."
        )
    return model


def extract_llm_text(content: object) -> str:
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
