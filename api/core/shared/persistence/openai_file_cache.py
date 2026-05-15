from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from aiocache import SimpleMemoryCache, cached
from loguru import logger

from api.config import get_settings

from .common import utc_now
from .documents import OpenAIFileCacheDocument


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


@cached(ttl=300, cache=SimpleMemoryCache, key_builder=_cache_key_builder)
async def load_cached_openai_file(
    *,
    provider_fingerprint: str,
    sha256: str,
) -> FileCacheEntry | None:
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
        return None
    return FileCacheEntry(file_id=doc.file_id, purpose=doc.purpose)


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
    try:
        await load_cached_openai_file.cache.delete(_memory_cache_key(provider_fingerprint, sha256))
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
    except Exception:
        logger.exception(
            "[OpenAIFileCacheStore] save failed provider={} sha256={} file_id={}",
            provider_fingerprint,
            sha256[:12],
            file_id,
        )


async def delete_cached_openai_file(*, provider_fingerprint: str, sha256: str) -> None:
    try:
        await load_cached_openai_file.cache.delete(_memory_cache_key(provider_fingerprint, sha256))
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
