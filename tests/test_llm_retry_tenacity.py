from api.shared import retry as retry_module


def test_llm_retry_call_uses_tenacity_backoff(monkeypatch):
    sleep_calls: list[float] = []
    attempts = 0

    monkeypatch.setattr(
        retry_module,
        "resolve_llm_retry_policy",
        lambda: (3, 2.5),
    )

    def record_sleep(seconds: float):
        sleep_calls.append(seconds)

    monkeypatch.setattr(retry_module.time, "sleep", record_sleep)

    def fn():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("try again")
        return "ok"

    assert retry_module.llm_retry_call(label="demo", fn=fn) == "ok"
    assert attempts == 3
    assert sleep_calls == [2.5, 5.0]
