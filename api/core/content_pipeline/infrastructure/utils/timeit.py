"""Timing utilities."""

from collections.abc import Callable
from functools import wraps
import time
from types import TracebackType
from typing import Any, Self

from loguru import logger


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Function {} failed",
                func.__name__,
                elapsed_seconds=round(elapsed, 2),
                error=str(e),
            )
            raise
        else:
            elapsed = time.time() - start_time
            logger.info(
                "Function {} completed",
                func.__name__,
                elapsed_seconds=round(elapsed, 2),
            )
            return result

    return wrapper


def atimeit(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Function {} failed",
                func.__name__,
                elapsed_seconds=round(elapsed, 2),
                error=str(e),
            )
            raise
        else:
            elapsed = time.time() - start_time
            logger.info(
                "Function {} completed",
                func.__name__,
                elapsed_seconds=round(elapsed, 2),
            )
            return result

    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "block") -> None:
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> Self:
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        start_time = self.start_time if self.start_time is not None else time.time()
        self.elapsed = time.time() - start_time

        if exc_type is None:
            logger.info(
                "Timer '{}' completed",
                self.name,
                elapsed_seconds=round(self.elapsed, 2),
            )
        else:
            logger.error(
                "Timer '{}' failed",
                self.name,
                elapsed_seconds=round(self.elapsed, 2),
                error=str(exc_val),
            )
