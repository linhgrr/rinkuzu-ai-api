"""Tests for the standardized retry layer in api.shared.retry.

Retry now lives in the LLM client (api.shared.llm), which composes the generic
``sync_retry`` / ``async_retry`` decorators below with ``is_retryable_llm_error``
+ ``resolve_llm_retry_policy``. These tests cover the primitives directly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import litellm
import pytest

from api.shared import retry as retry_module
from api.shared.retry import (
    async_retry,
    is_retryable_llm_error,
    resolve_llm_retry_policy,
    sync_retry,
)


def test_is_retryable_llm_error_retries_transient_or_unknown_exceptions():
    assert is_retryable_llm_error(ValueError("bad")) is True
    assert is_retryable_llm_error(RuntimeError("x")) is True
    assert (
        is_retryable_llm_error(
            litellm.RateLimitError("limited", llm_provider="openai", model="gpt")
        )
        is True
    )
    assert is_retryable_llm_error(KeyboardInterrupt()) is False  # not an Exception


def test_is_retryable_llm_error_excludes_non_retryable_litellm_errors():
    response = httpx.Response(400, request=httpx.Request("POST", "https://llm.test"))
    non_retryable = [
        litellm.AuthenticationError("bad key", llm_provider="openai", model="gpt"),
        litellm.BadRequestError("bad request", model="gpt", llm_provider="openai"),
        litellm.NotFoundError("missing model", model="gpt", llm_provider="openai"),
        litellm.PermissionDeniedError(
            "forbidden",
            llm_provider="openai",
            model="gpt",
            response=response,
        ),
        litellm.ContentPolicyViolationError("blocked", model="gpt", llm_provider="openai"),
        litellm.UnprocessableEntityError(
            "unprocessable",
            model="gpt",
            llm_provider="openai",
            response=response,
        ),
    ]

    assert all(is_retryable_llm_error(exc) is False for exc in non_retryable)


def test_resolve_llm_retry_policy_reads_settings(monkeypatch):
    import api.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(llm_retry_attempts=5, llm_retry_backoff_sec=2.5),
    )
    assert resolve_llm_retry_policy() == (5, 2.5)


def test_resolve_llm_retry_policy_clamps(monkeypatch):
    import api.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(llm_retry_attempts=0, llm_retry_backoff_sec=-1.0),
    )
    assert resolve_llm_retry_policy() == (1, 0.0)


def test_sync_retry_succeeds_after_retries():
    attempts = {"n": 0}

    @sync_retry(
        label="demo",
        max_attempts=3,
        base_delay_sec=0.0,
        retry_on=is_retryable_llm_error,
    )
    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("try again")
        return "ok"

    assert fn() == "ok"
    assert attempts["n"] == 3


def test_sync_retry_reraises_after_exhaustion():
    @sync_retry(
        label="demo",
        max_attempts=2,
        base_delay_sec=0.0,
        retry_on=is_retryable_llm_error,
    )
    def fn() -> str:
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        fn()


def test_sync_retry_uses_exponential_backoff(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", sleep_calls.append)
    attempts = {"n": 0}

    @sync_retry(
        label="demo",
        max_attempts=3,
        base_delay_sec=2.5,
        retry_on=is_retryable_llm_error,
    )
    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("try again")
        return "ok"

    assert fn() == "ok"
    # wait_exponential(multiplier=2.5): 2.5*2^0, 2.5*2^1 = 5.0
    assert sleep_calls == [2.5, 5.0]


def test_async_retry_retries_broadly_then_succeeds():
    attempts = {"n": 0}

    @async_retry(
        label="async demo",
        max_attempts=3,
        base_delay_sec=0.0,
        retry_on=is_retryable_llm_error,
    )
    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient json")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert attempts["n"] == 3


def test_async_retry_reraises_after_exhaustion():
    @async_retry(
        label="async demo",
        max_attempts=2,
        base_delay_sec=0.0,
        retry_on=is_retryable_llm_error,
    )
    async def boom() -> str:
        raise ValueError("permanently broken")

    with pytest.raises(ValueError, match="permanently broken"):
        asyncio.run(boom())
