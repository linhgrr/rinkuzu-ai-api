from types import SimpleNamespace

from api.core.shared import llm_usage


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
