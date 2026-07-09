"""Tests for the standardized LLM retry layer in api.shared.retry."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from api.shared import retry as retry_module
from api.shared.retry import (
    is_retryable_llm_error,
    llm_async_retry,
    llm_retry_call,
    resolve_llm_retry_policy,
)


def test_is_retryable_llm_error_broad():
    assert is_retryable_llm_error(ValueError("bad")) is True
    assert is_retryable_llm_error(RuntimeError("x")) is True
    assert is_retryable_llm_error(KeyboardInterrupt()) is False  # not an Exception


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


def test_llm_retry_call_succeeds_after_retries(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (3, 0.0))
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("try again")
        return "ok"

    assert llm_retry_call(label="demo", fn=fn) == "ok"
    assert attempts["n"] == 3


def test_llm_retry_call_returns_on_exhausted(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (2, 0.0))

    def fn():
        raise RuntimeError("always fails")

    result = llm_retry_call(
        label="demo",
        fn=fn,
        on_exhausted=lambda: "fallback",
    )
    assert result == "fallback"


def test_llm_retry_call_raises_runtimeerror_without_fallback(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (2, 0.0))

    def fn():
        raise ValueError("provider down")

    with pytest.raises(RuntimeError, match="demo is temporarily unavailable"):
        llm_retry_call(label="demo", fn=fn)


def test_llm_retry_call_uses_backoff(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (3, 2.5))
    sleep_calls: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", sleep_calls.append)
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("try again")
        return "ok"

    assert llm_retry_call(label="demo", fn=fn) == "ok"
    assert sleep_calls == [2.5, 5.0]


def test_llm_async_retry_retries_broadly_then_succeeds(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (3, 0.0))
    attempts = {"n": 0}

    @llm_async_retry(label="async demo")
    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient json")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert attempts["n"] == 3


def test_llm_async_retry_raises_runtimeerror_on_exhaustion(monkeypatch):
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (2, 0.0))

    @llm_async_retry(label="async demo")
    async def boom() -> str:
        raise ValueError("permanently broken")

    with pytest.raises(RuntimeError, match="async demo is temporarily unavailable"):
        asyncio.run(boom())
