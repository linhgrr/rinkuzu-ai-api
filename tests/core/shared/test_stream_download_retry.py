"""Tests: stream_download retries transient errors; never retries UnsafeURLError."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self
from unittest.mock import patch

import httpx
import pytest

from api.shared.url_fetch import UnsafeURLError, stream_download

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAFE_URL = "https://example.com/file.pdf"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# Shared state written by _FakeStream so tests can inspect call counts.
_stream_enter_count = 0
_stream_payload: bytes = b""


# ---------------------------------------------------------------------------
# Fake httpx client / stream (module-level to keep complexity down)
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async context-manager that simulates a streaming httpx response."""

    def __init__(self, *, fail_first: bool) -> None:
        self._fail_first = fail_first

    async def __aenter__(self) -> Self:
        global _stream_enter_count
        _stream_enter_count += 1
        if self._fail_first and _stream_enter_count == 1:
            raise httpx.ConnectError("simulated reset")
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False

    @property
    def status_code(self) -> int:
        return 200

    @property
    def headers(self) -> dict:
        return {}

    def raise_for_status(self) -> None:
        pass

    async def aiter_bytes(self, chunk_size: int = 65536):
        yield _stream_payload


class _FakeClient:
    """Async context-manager that mimics httpx.AsyncClient."""

    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False

    def stream(self, method: str, url: str) -> _FakeStream:
        return _FakeStream(fail_first=True)


# ---------------------------------------------------------------------------
# Case (a): transient error on first attempt, success on second
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_download_retries_on_transient_error(tmp_path: Path) -> None:
    """A ConnectError on the first attempt is retried; the file is written on the second."""
    global _stream_enter_count, _stream_payload
    _stream_enter_count = 0
    _stream_payload = b"PDF content here"

    dest = tmp_path / "out.pdf"

    with patch("api.shared.url_fetch.httpx.AsyncClient", _FakeClient):
        bytes_written = await stream_download(_SAFE_URL, dest, max_bytes=_MAX_BYTES)

    assert bytes_written == len(_stream_payload)
    assert dest.exists()
    assert dest.read_bytes() == _stream_payload
    # The stream was entered more than once (one failure + one success)
    assert _stream_enter_count >= 2, f"Expected at least 2 attempts, got {_stream_enter_count}"


# ---------------------------------------------------------------------------
# Case (b): UnsafeURLError is raised exactly once, zero retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_download_does_not_retry_unsafe_url(tmp_path: Path) -> None:
    """UnsafeURLError from validate_download_url must propagate immediately, no retries."""
    dest = tmp_path / "out.pdf"

    # http:// scheme is blocked by validate_download_url — no network call needed.
    unsafe_url = "http://example.com/file.pdf"

    validate_call_count = 0

    import api.shared.url_fetch as _mod

    original_validate = _mod.validate_download_url

    def _counting_validate(url: str, allowlist: list | None = None) -> None:
        nonlocal validate_call_count
        validate_call_count += 1
        original_validate(url, allowlist=allowlist)

    with (
        patch.object(_mod, "validate_download_url", _counting_validate),
        pytest.raises(UnsafeURLError),
    ):
        await stream_download(unsafe_url, dest, max_bytes=_MAX_BYTES)

    # Validation ran exactly once — no retry loop touched it.
    assert validate_call_count == 1, (
        f"validate_download_url should be called exactly once; got {validate_call_count}"
    )
    assert not dest.exists(), "No file should be created when URL is unsafe"
