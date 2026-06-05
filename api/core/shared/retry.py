"""Transient-error classification + async retry built on tenacity."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
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


def async_transient_retry(
    *,
    label: str,
    max_attempts: int,
    base_delay_sec: float,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Retry an async callable only on transient errors, with exp backoff."""

    def decorator(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @wraps(fn)
        async def wrapped(*args: object, **kwargs: object) -> _T:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(max(1, max_attempts)),
                wait=wait_exponential(multiplier=max(0.0, base_delay_sec), max=60),
                retry=retry_if_exception(is_transient_error),
                reraise=True,
                sleep=asyncio.sleep,
                before_sleep=lambda rs: logger.warning(
                    "[retry] {} attempt {} failed: {}",
                    label,
                    rs.attempt_number,
                    rs.outcome.exception() if rs.outcome else None,
                ),
            )

            async def _call() -> _T:
                return await fn(*args, **kwargs)

            return await retrying(_call)

        return wrapped

    return decorator
