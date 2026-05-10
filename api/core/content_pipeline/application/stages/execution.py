"""Execution helpers shared by blocking content pipeline stages."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.domain.errors import PipelineStageTimeoutError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")
_T = TypeVar("_T")


def resolve_timeout_policy() -> tuple[float | None, float | None]:
    """Return normalized job and stage timeout values from settings."""
    settings = get_settings()
    return (
        _normalize_timeout(settings.content_pipeline_job_timeout_sec),
        _normalize_timeout(settings.content_pipeline_stage_timeout_sec),
    )


async def run_blocking_stage(
    func: Callable[..., T],
    *args: Any,
    stage_name: str,
    timeout_sec: float | None = None,
    **kwargs: Any,
) -> T:
    """Run a blocking stage function in a worker thread with optional timeout."""
    _, default_stage_timeout = resolve_timeout_policy()
    effective_timeout = timeout_sec if timeout_sec is not None else default_stage_timeout

    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)
    future = loop.run_in_executor(None, call)

    if effective_timeout is None:
        return await future

    try:
        return await asyncio.wait_for(future, timeout=effective_timeout)
    except TimeoutError as exc:
        raise PipelineStageTimeoutError(stage_name, effective_timeout) from exc


def _normalize_timeout(raw_value: float | None) -> float | None:
    if raw_value is None:
        return None
    value = float(raw_value)
    return value if value > 0 else None


async def safe_run(
    fn: Callable[[], Awaitable[_T]],
    *,
    fail_message: str,
    fallback: _T | None = None,
) -> _T | None:
    """Best-effort async execution. Logs exception and returns *fallback* on failure."""
    try:
        return await fn()
    except Exception:
        logger.exception("[Pipeline] {}", fail_message)
        return fallback
