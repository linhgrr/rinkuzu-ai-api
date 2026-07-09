import httpx
import pytest

from api.shared.retry import async_transient_retry, is_transient_error


def test_classifier_transient_vs_permanent():
    assert is_transient_error(httpx.ConnectError("x")) is True
    assert is_transient_error(TimeoutError()) is True
    assert (
        is_transient_error(TimeoutError()) is True
    )  # asyncio.TimeoutError == TimeoutError on 3.11+
    assert is_transient_error(ValueError("bad input")) is False


def test_classifier_http_status():
    req = httpx.Request("GET", "http://x")
    assert (
        is_transient_error(
            httpx.HTTPStatusError("e", request=req, response=httpx.Response(500, request=req))
        )
        is True
    )
    assert (
        is_transient_error(
            httpx.HTTPStatusError("e", request=req, response=httpx.Response(400, request=req))
        )
        is False
    )
    assert (
        is_transient_error(
            httpx.HTTPStatusError("e", request=req, response=httpx.Response(429, request=req))
        )
        is True
    )


@pytest.mark.asyncio
async def test_async_transient_retry_retries_then_succeeds():
    calls = {"n": 0}

    @async_transient_retry(label="probe", max_attempts=3, base_delay_sec=0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_async_transient_retry_does_not_retry_permanent():
    calls = {"n": 0}

    @async_transient_retry(label="probe", max_attempts=3, base_delay_sec=0)
    async def boom() -> str:
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        await boom()
    assert calls["n"] == 1
