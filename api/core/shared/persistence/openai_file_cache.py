from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import time
from typing import Any

from loguru import logger

from api.config import get_settings

from .common import utc_now
from .documents import OpenAIFileCacheDocument

try:
    from aiocache import SimpleMemoryCache as _AiocacheSimpleMemoryCache
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in minimal test envs
    logger.warning("[OpenAIFileCacheStore] aiocache not installed; using local fallback cache")

    class _FallbackSimpleMemoryCache:
        def __init__(self) -> None:
            self._store: dict[str, tuple[float | None, object]] = {}

        async def get(self, key: str) -> object | None:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and expires_at <= time.monotonic():
                self._store.pop(key, None)
                return None
            return value

        async def set(self, key: str, value: object, ttl: int | None = None) -> None:
            expires_at = time.monotonic() + ttl if ttl else None
            self._store[key] = (expires_at, value)

        async def delete(self, key: str) -> None:
            self._store.pop(key, None)

    _MemoryCacheImpl: type[Any] = _FallbackSimpleMemoryCache
else:
    _MemoryCacheImpl = _AiocacheSimpleMemoryCache


_MEMORY_CACHE = _MemoryCacheImpl()
_MEMORY_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class FileCacheEntry:
    file_id: str
    purpose: str


def _memory_cache_key(provider_fingerprint: str, sha256: str) -> str:
    return f"{provider_fingerprint}:{sha256}"


def _cache_key_builder(
    _func,
    *,
    provider_fingerprint: str,
    sha256: str,
) -> str:
    return _memory_cache_key(provider_fingerprint, sha256)


async def load_cached_openai_file(
    *,
    provider_fingerprint: str,
    sha256: str,
) -> FileCacheEntry | None:
    cache_key = _memory_cache_key(provider_fingerprint, sha256)
    cached_entry = await _MEMORY_CACHE.get(cache_key)
    if isinstance(cached_entry, FileCacheEntry):
        return cached_entry
    if cached_entry is None:
        cached_miss = False
    else:
        cached_miss = bool(isinstance(cached_entry, dict) and cached_entry.get("missing") is True)
        if cached_miss:
            return None

    try:
        doc = await OpenAIFileCacheDocument.find_one(
            OpenAIFileCacheDocument.provider_fingerprint == provider_fingerprint,
            OpenAIFileCacheDocument.sha256 == sha256,
        )
    except Exception:
        logger.exception(
            "[OpenAIFileCacheStore] load failed provider={} sha256={}",
            provider_fingerprint,
            sha256[:12],
        )
        return None
    if doc is None:
        await _MEMORY_CACHE.set(cache_key, {"missing": True}, ttl=_MEMORY_CACHE_TTL_SECONDS)
        return None
    entry = FileCacheEntry(file_id=doc.file_id, purpose=doc.purpose)
    await _MEMORY_CACHE.set(cache_key, entry, ttl=_MEMORY_CACHE_TTL_SECONDS)
    return entry


async def save_cached_openai_file(
    *,
    provider_fingerprint: str,
    sha256: str,
    file_id: str,
    purpose: str,
) -> None:
    ttl_hours = max(1, get_settings().content_pipeline_file_cache_ttl_hours)
    now = utc_now()
    expires_at = now + timedelta(hours=ttl_hours)
    cache_key = _memory_cache_key(provider_fingerprint, sha256)
    try:
        await _MEMORY_CACHE.delete(cache_key)
        existing = await OpenAIFileCacheDocument.find_one(
            OpenAIFileCacheDocument.provider_fingerprint == provider_fingerprint,
            OpenAIFileCacheDocument.sha256 == sha256,
        )
        if existing is None:
            await OpenAIFileCacheDocument(
                provider_fingerprint=provider_fingerprint,
                sha256=sha256,
                file_id=file_id,
                purpose=purpose,
                created_at=now,
                expires_at=expires_at,
            ).insert()
        else:
            existing.file_id = file_id
            existing.purpose = purpose
            existing.expires_at = expires_at
            await existing.replace()
        await _MEMORY_CACHE.set(
            cache_key,
            FileCacheEntry(file_id=file_id, purpose=purpose),
            ttl=_MEMORY_CACHE_TTL_SECONDS,
        )
    except Exception:
        logger.exception(
            "[OpenAIFileCacheStore] save failed provider={} sha256={} file_id={}",
            provider_fingerprint,
            sha256[:12],
            file_id,
        )


async def delete_cached_openai_file(*, provider_fingerprint: str, sha256: str) -> None:
    cache_key = _memory_cache_key(provider_fingerprint, sha256)
    try:
        await _MEMORY_CACHE.delete(cache_key)
        await OpenAIFileCacheDocument.find(
            OpenAIFileCacheDocument.provider_fingerprint == provider_fingerprint,
            OpenAIFileCacheDocument.sha256 == sha256,
        ).delete()
    except Exception:
        logger.exception(
            "[OpenAIFileCacheStore] delete failed provider={} sha256={}",
            provider_fingerprint,
            sha256[:12],
        )
