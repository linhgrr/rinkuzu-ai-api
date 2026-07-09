"""Canonical retry layer: transient-error classification + sync/async retry.

This is the single source of truth for retry behaviour across the codebase.
It exposes the core decorators ``sync_retry`` / ``async_retry`` — generic
tenacity-backed retry with a configurable ``retry_on`` predicate and
``[retry]`` logging. The LLM client (``api.shared.llm``) composes these with
``is_retryable_llm_error`` + ``resolve_llm_retry_policy`` so every LLM call
retries transient failures by default; call sites don't wrap anything.
"""

from __future__ import annotations

import asyncio
from functools import wraps
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_T = TypeVar("_T")

_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def is_transient_error(exc: BaseException) -> bool:
    """True for errors worth retrying (network/timeout/5xx/429)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True
    return bool(isinstance(exc, TimeoutError))


def is_retryable_llm_error(exc: BaseException) -> bool:
    """LLM calls: failures are predominantly transient (provider 5xx, timeouts,
    rate limits, malformed streams). Retry broadly on Exception."""
    return isinstance(exc, Exception)


def resolve_llm_retry_policy() -> tuple[int, float]:
    """(max_attempts, base_delay_sec) from settings.llm_retry_attempts / llm_retry_backoff_sec."""
    from api.config import get_settings

    s = get_settings()
    return max(1, int(s.llm_retry_attempts)), max(0.0, float(s.llm_retry_backoff_sec))


def _make_before_sleep(label: str) -> Callable[[Any], None]:
    def _before_sleep(retry_state: Any) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "[retry] {} attempt {} failed: {}",
            label,
            retry_state.attempt_number,
            exc,
        )

    return _before_sleep


def async_retry(
    *,
    label: str,
    max_attempts: int,
    base_delay_sec: float,
    retry_on: Callable[[BaseException], bool] = is_transient_error,
    max_wait_sec: float = 60,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Retry an async callable on ``retry_on`` errors, with exp backoff."""

    def decorator(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @wraps(fn)
        async def wrapped(*args: object, **kwargs: object) -> _T:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(max(1, max_attempts)),
                wait=wait_exponential(multiplier=max(0.0, base_delay_sec), max=max_wait_sec),
                retry=retry_if_exception(retry_on),
                reraise=True,
                sleep=asyncio.sleep,
                before_sleep=_make_before_sleep(label),
            )

            async def _call() -> _T:
                return await fn(*args, **kwargs)

            return cast("_T", await retrying(_call))

        return wrapped

    return decorator


def sync_retry(
    *,
    label: str,
    max_attempts: int,
    base_delay_sec: float,
    retry_on: Callable[[BaseException], bool] = is_transient_error,
    max_wait_sec: float = 60,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Retry a sync callable on ``retry_on`` errors, with exp backoff."""

    def decorator(fn: Callable[..., _T]) -> Callable[..., _T]:
        @wraps(fn)
        def wrapped(*args: object, **kwargs: object) -> _T:
            retrying = Retrying(
                stop=stop_after_attempt(max(1, max_attempts)),
                wait=wait_exponential(multiplier=max(0.0, base_delay_sec), max=max_wait_sec),
                retry=retry_if_exception(retry_on),
                reraise=True,
                sleep=time.sleep,
                before_sleep=_make_before_sleep(label),
            )

            def _call() -> _T:
                return fn(*args, **kwargs)

            return retrying(_call)

        return wrapped

    return decorator


# `async_transient_retry` kept as an alias for the URL-fetch path, whose
# semantics (transient-only, async, exp backoff) are exactly `async_retry`.
async_transient_retry = async_retry
