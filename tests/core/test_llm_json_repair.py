"""Tests for json_repair integration in generate_structured / agenerate_structured."""

from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from api.shared import llm as llm_module
from api.shared.llm import invoke_structured_completion


class ExerciseSchema(BaseModel):
    exercise_type: str
    question: str


FAKE_SETTINGS = SimpleNamespace(
    llm_base_url="https://api.example.com",
    llm_api_key="test-key",  # pragma: allowlist secret
    llm_model="model-x",
    llm_timeout_sec=30,
    exercise_llm_model=None,
    llm_retry_attempts=1,
    llm_retry_backoff_sec=0.0,
)


def _make_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _patch(monkeypatch, content: str):
    monkeypatch.setattr(llm_module, "completion", lambda **_: _make_response(content))
    monkeypatch.setattr(llm_module, "get_settings", lambda: FAKE_SETTINGS)


def _call():
    return invoke_structured_completion(
        schema=ExerciseSchema,
        model="model-x",
        messages=[{"role": "user", "content": "generate"}],
    )


def test_plain_json(monkeypatch):
    _patch(monkeypatch, '{"exercise_type": "fill_blank", "question": "Q?"}')
    result = _call()
    assert result == ExerciseSchema(exercise_type="fill_blank", question="Q?")


def test_markdown_json_fence(monkeypatch):
    content = '```json\n{"exercise_type": "fill_blank", "question": "Q?"}\n```'
    _patch(monkeypatch, content)
    result = _call()
    assert result == ExerciseSchema(exercise_type="fill_blank", question="Q?")


def test_markdown_fence_no_lang(monkeypatch):
    content = '```\n{"exercise_type": "mcq", "question": "Q2?"}\n```'
    _patch(monkeypatch, content)
    result = _call()
    assert result == ExerciseSchema(exercise_type="mcq", question="Q2?")


def test_trailing_comma(monkeypatch):
    _patch(monkeypatch, '{"exercise_type": "mcq", "question": "Q?",}')
    result = _call()
    assert result == ExerciseSchema(exercise_type="mcq", question="Q?")


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_or_whitespace_raises(monkeypatch, content):
    _patch(monkeypatch, content)
    with pytest.raises(TypeError, match="empty structured output"):
        _call()


def test_client_retries_transient_failure_then_succeeds(monkeypatch):
    """Retry is a client default: a transient failure is retried automatically,
    with no per-call-site wrapper. This is the attempt-count contract that used
    to live in exercise_gen / extract_chain tests before centralization.
    """
    # llm.py binds resolve_llm_retry_policy at import — patch it there, not in
    # retry_module, or the client keeps using the real settings-derived policy.
    monkeypatch.setattr(llm_module, "resolve_llm_retry_policy", lambda: (3, 0.0))
    monkeypatch.setattr(llm_module, "get_settings", lambda: FAKE_SETTINGS)

    calls = {"n": 0}

    def _flaky_completion(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient provider failure")
        return _make_response('{"exercise_type": "mcq", "question": "Q?"}')

    monkeypatch.setattr(llm_module, "completion", _flaky_completion)

    result = _call()

    assert calls["n"] == 2
    assert result == ExerciseSchema(exercise_type="mcq", question="Q?")


def test_client_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm_module, "resolve_llm_retry_policy", lambda: (2, 0.0))
    monkeypatch.setattr(llm_module, "get_settings", lambda: FAKE_SETTINGS)

    calls = {"n": 0}

    def _always_fail(**_):
        calls["n"] += 1
        raise RuntimeError("permanent provider failure")

    monkeypatch.setattr(llm_module, "completion", _always_fail)

    with pytest.raises(RuntimeError, match="permanent provider failure"):
        _call()
    assert calls["n"] == 2
