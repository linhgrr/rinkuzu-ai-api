"""Execution helpers shared by blocking content pipeline stages."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import import_module
import multiprocessing
import os
import time
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.domain.errors import PipelineStageTimeoutError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")
_T = TypeVar("_T")

# Dedicated thread pool for pipeline CPU/blocking-I/O work.
# Separate from FastAPI's default executor so long-running pipeline stages
# (PDF rendering, embedding, ChromaDB, sync S3) never starve other endpoints.
_PIPELINE_MAX_WORKERS = int(
    os.environ.get("PIPELINE_THREAD_POOL_SIZE", max(4, (os.cpu_count() or 2)))
)
_pipeline_executor: ThreadPoolExecutor | None = None
_inflight_blocking_futures: set[asyncio.Future[Any]] = set()


def get_pipeline_executor() -> ThreadPoolExecutor:
    global _pipeline_executor
    if _pipeline_executor is None:
        _pipeline_executor = ThreadPoolExecutor(
            max_workers=_PIPELINE_MAX_WORKERS,
            thread_name_prefix="pipeline-",
        )
    return _pipeline_executor


async def run_blocking_stage(
    func: Callable[..., T],
    *args: Any,
    stage_name: str,
    timeout_sec: float | None = None,
    **kwargs: Any,
) -> T:
    """Run a blocking stage function in the pipeline thread pool with optional timeout."""
    _, default_stage_timeout = resolve_timeout_policy()
    effective_timeout = timeout_sec if timeout_sec is not None else default_stage_timeout

    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)
    future = loop.run_in_executor(get_pipeline_executor(), call)
    _inflight_blocking_futures.add(future)
    future.add_done_callback(_inflight_blocking_futures.discard)

    if effective_timeout is None:
        return await future

    try:
        return await asyncio.wait_for(future, timeout=effective_timeout)
    except asyncio.CancelledError:
        future.cancel()
        raise
    except TimeoutError as exc:
        future.cancel()
        raise PipelineStageTimeoutError(stage_name, float(effective_timeout or 0.0)) from exc


def resolve_timeout_policy() -> tuple[float | None, float | None]:
    """Return normalized job and stage timeout values from settings."""
    settings = get_settings()
    return (
        _normalize_timeout(settings.content_pipeline_job_timeout_sec),
        _normalize_timeout(settings.content_pipeline_stage_timeout_sec),
    )


def _normalize_timeout(raw_value: float | None) -> float | None:
    if raw_value is None:
        return None
    value = float(raw_value)
    return value if value > 0 else None


def shutdown_pipeline_executor(*, wait: bool = True, cancel_futures: bool = False) -> None:
    """Shut down the dedicated pipeline executor."""
    global _pipeline_executor
    if _pipeline_executor is None:
        return
    _pipeline_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    _pipeline_executor = None
    _inflight_blocking_futures.clear()


def _resolve_target(target_path: str) -> Any:
    module_name, _, attr_path = target_path.partition(":")
    if not module_name or not attr_path:
        raise ValueError(f"Invalid process target path: {target_path}")
    module = import_module(module_name)
    target = module
    for attr in attr_path.split("."):
        target = getattr(target, attr)
    return target


def _process_stage_entrypoint(
    conn: multiprocessing.connection.Connection,
    target_path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        target = _resolve_target(target_path)
        result = target(*args, **kwargs)
        conn.send(("ok", result))
    except BaseException as exc:  # pragma: no cover - subprocess error propagation
        conn.send(
            (
                "err",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        )
    finally:
        conn.close()


async def run_process_stage(
    target_path: str,
    *args: Any,
    stage_name: str,
    timeout_sec: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run a top-level callable in an isolated subprocess so timeout can hard-stop it."""
    _, default_stage_timeout = resolve_timeout_policy()
    effective_timeout = timeout_sec if timeout_sec is not None else default_stage_timeout

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_process_stage_entrypoint,
        args=(child_conn, target_path, args, kwargs),
        daemon=True,
    )
    process.start()
    child_conn.close()

    try:
        if effective_timeout is None:
            while not await asyncio.to_thread(parent_conn.poll, 0.1):
                await asyncio.sleep(0)
        else:
            deadline = time.monotonic() + effective_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                if await asyncio.to_thread(parent_conn.poll, min(0.1, remaining)):
                    break
                await asyncio.sleep(0)
        status, payload = parent_conn.recv()
    except asyncio.CancelledError:
        if process.is_alive():
            process.kill() if hasattr(process, "kill") else process.terminate()
        await asyncio.to_thread(process.join, 1.0)
        raise
    except TimeoutError as exc:
        if process.is_alive():
            process.kill() if hasattr(process, "kill") else process.terminate()
        await asyncio.to_thread(process.join, 1.0)
        raise PipelineStageTimeoutError(stage_name, float(effective_timeout or 0.0)) from exc
    except EOFError as exc:
        await asyncio.to_thread(process.join, 1.0)
        raise RuntimeError(f"{stage_name} failed in isolated process: no result returned") from exc
    finally:
        parent_conn.close()

    await asyncio.to_thread(process.join, 1.0)
    if status == "ok":
        return payload

    error_type = payload.get("type", "ProcessError")
    error_message = payload.get("message", "Unknown process failure")
    raise RuntimeError(f"{stage_name} failed in isolated process [{error_type}]: {error_message}")


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
