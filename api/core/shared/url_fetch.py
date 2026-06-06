"""Safe URL fetching: allowlist, private-IP block, size cap, timeout."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import aiofiles
import httpx

from api.config import get_settings
from api.core.shared.retry import async_transient_retry

if TYPE_CHECKING:
    from pathlib import Path

ALLOWED_SCHEMES = {"https"}
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


class UnsafeURLError(ValueError):
    """Raised when a URL fails safety validation."""


def _resolve_ips(host: str) -> list[str]:
    try:
        return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]
    except socket.gaierror as exc:
        raise UnsafeURLError(f"DNS resolution failed for '{host}'") from exc


def _is_private_host(host: str) -> bool:
    for raw_ip in _resolve_ips(host):
        ip = ipaddress.ip_address(raw_ip)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def validate_download_url(url: str, allowlist: list[str] | None = None) -> None:
    """Raise UnsafeURLError if *url* must not be fetched."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Scheme '{parsed.scheme}' is not allowed (only https)")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no hostname")
    if allowlist and host not in allowlist:
        raise UnsafeURLError(f"Host '{host}' is not in the download allowlist")
    if _is_private_host(host):
        raise UnsafeURLError(f"Host '{host}' resolves to a private/reserved address")


async def _fetch_and_write(url: str, dest_path: Path, *, max_bytes: int) -> int:
    """Perform the actual HTTP GET and write bytes to *dest_path*.

    This inner coroutine is wrapped with transient-retry logic in
    ``stream_download``.  Any partial file from a failed attempt is removed
    before the next retry so the size cap remains accurate.
    """
    # Remove any partial file left by a previous failed attempt.
    await asyncio.to_thread(dest_path.unlink, missing_ok=True)

    bytes_written = 0
    async with (
        httpx.AsyncClient(
            timeout=_DOWNLOAD_TIMEOUT,
            follow_redirects=False,
        ) as client,
        client.stream("GET", url) as resp,
    ):
        resp.raise_for_status()

        content_length = resp.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise UnsafeURLError(f"Content-Length {content_length} exceeds {max_bytes}-byte limit")

        async with aiofiles.open(dest_path, "wb") as fh:
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    await fh.close()
                    await asyncio.to_thread(dest_path.unlink, missing_ok=True)
                    raise UnsafeURLError(f"Download exceeded {max_bytes}-byte limit")
                await fh.write(chunk)

    return bytes_written


async def stream_download(
    url: str,
    dest_path: Path,
    *,
    max_bytes: int,
    allowlist: list[str] | None = None,
) -> int:
    """Download *url* to *dest_path*, enforcing safety checks and a size cap.

    Returns the number of bytes written.
    Raises UnsafeURLError for policy violations, httpx.HTTPError for HTTP failures.

    Transient network errors (connection reset, 5xx, timeout) are retried with
    exponential backoff.  SSRF validation happens before any retry, so
    ``UnsafeURLError`` is always raised immediately without retrying.
    """
    # SSRF / policy validation — runs once, never inside the retry loop.
    validate_download_url(url, allowlist=allowlist)

    settings = get_settings()

    @async_transient_retry(
        label="download source",
        max_attempts=settings.content_pipeline_llm_retry_attempts,
        base_delay_sec=settings.content_pipeline_llm_retry_backoff_sec,
    )
    async def _retryable() -> int:
        return await _fetch_and_write(url, dest_path, max_bytes=max_bytes)

    return await _retryable()
