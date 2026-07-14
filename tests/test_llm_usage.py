from types import SimpleNamespace

import pytest

from api.shared import llm_usage


def test_compute_cost_flash_1m_in_1m_out():
    # flash: 0.14 in + 0.28 out per 1M
    assert llm_usage.compute_cost_usd("deepseek-v4-flash", 1_000_000, 1_000_000) == 0.42


def test_compute_cost_pro_matched_by_substring():
    # pro: 0.435 in + 0.87 out per 1M
    assert llm_usage.compute_cost_usd("deepseek-v4-pro", 1_000_000, 1_000_000) == 1.305


def test_compute_cost_unknown_model_falls_back_to_flash():
    assert llm_usage.compute_cost_usd("some-other-model", 1_000_000, 0) == 0.14


def test_extract_usage_from_dict():
    usage = llm_usage.extract_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_extract_usage_from_object():
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=7, total_tokens=10)
    )
    usage = llm_usage.extract_usage(resp)
    assert usage == {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}


def test_extract_usage_missing_returns_none():
    assert llm_usage.extract_usage({"no_usage": 1}) is None
    assert llm_usage.extract_usage(SimpleNamespace()) is None


@pytest.mark.anyio
async def test_record_llm_usage_persists_action(monkeypatch):
    """The feature label passed by the caller must reach the persisted record.

    Guards the regression where action was read from an unset ContextVar and
    every record landed as (unknown).
    """
    captured: dict[str, object] = {}

    class _FakeDoc:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def insert(self):
            return None

    monkeypatch.setattr(llm_usage.mongo_store, "is_available", lambda: True)
    monkeypatch.setattr(llm_usage, "LlmUsageDocument", _FakeDoc)

    await llm_usage.record_llm_usage(
        model="deepseek-v4-flash",
        provider="deepseek",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        action=llm_usage.LlmAction.ADAPTIVE_EXERCISE,
    )

    assert captured["action"] == "adaptive_exercise"
    assert captured["input_tokens"] == 10


@pytest.mark.anyio
async def test_record_llm_usage_persists_current_user(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeDoc:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def insert(self):
            return None

    monkeypatch.setattr(llm_usage.mongo_store, "is_available", lambda: True)
    monkeypatch.setattr(llm_usage, "LlmUsageDocument", _FakeDoc)
    token = llm_usage.current_user_id.set("user-1")
    try:
        await llm_usage.record_llm_usage(
            model="deepseek-v4-flash",
            provider="deepseek",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            action=llm_usage.LlmAction.ADAPTIVE_EXERCISE,
        )
    finally:
        llm_usage.current_user_id.reset(token)

    assert captured["user_id"] == "user-1"


@pytest.mark.anyio
async def test_record_usage_async_creates_task_on_running_loop(monkeypatch):
    """LLM calls are async-native now: usage recording fires from within an async
    context via _record_usage_async, awaited on the same running loop the Mongo
    client is bound to — no thread-hop, no cross-loop scheduling.
    """
    from api.shared import llm as llm_module

    recorded: dict[str, object] = {}

    async def _fake_record(**kwargs: object) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(llm_module, "record_llm_usage", _fake_record)

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    )
    config = SimpleNamespace(model="m", custom_llm_provider="deepseek")

    await llm_module._record_usage_async(config, response, "adaptive_exercise")

    assert recorded["action"] == "adaptive_exercise"
    assert recorded["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
