from types import SimpleNamespace

from api.core.shared import llm as llm_module


def test_with_llm_retry_uses_tenacity_backoff(monkeypatch):
    sleep_calls: list[float] = []
    attempts = 0

    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(llm_retry_attempts=3, llm_retry_backoff_sec=2.5),
    )

    def record_sleep(seconds: float):
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_module.time, "sleep", record_sleep)

    def fn():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("try again")
        return "ok"

    assert llm_module.with_llm_retry(label="demo", fn=fn) == "ok"
    assert attempts == 3
    assert sleep_calls == [2.5, 5.0]
