"""
repositories/base.py — Shared helpers for Mongo-backed repositories.
"""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from loguru import logger


T = TypeVar("T")


class MongoRepository:
    """Small async helper for repository methods with stable fallback values."""

    def __init__(self, db):
        self._db = db

    async def _run_or_default(
        self,
        operation: str,
        default: T,
        func: Callable[[], Awaitable[T]],
    ) -> T:
        try:
            return await func()
        except Exception:
            logger.exception("[{}] {} failed", self.__class__.__name__, operation)
            return default
