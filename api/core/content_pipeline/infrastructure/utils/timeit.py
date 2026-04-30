"""Timing utilities."""

from collections.abc import Callable
from functools import wraps
import time
from typing import Any

from loguru import logger


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            logger.info(
                f"Function {func.__name__} completed",
                elapsed_seconds=round(elapsed, 2),
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed",
                elapsed_seconds=round(elapsed, 2),
                error=str(e),
            )
            raise

    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "block"):
        """
        Initialize timer.

        Args:
            name: Name of the timed block
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log."""
        self.elapsed = time.time() - self.start_time

        if exc_type is None:
            logger.info(
                f"Timer '{self.name}' completed",
                elapsed_seconds=round(self.elapsed, 2),
            )
        else:
            logger.error(
                f"Timer '{self.name}' failed",
                elapsed_seconds=round(self.elapsed, 2),
                error=str(exc_val),
            )
