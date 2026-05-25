"""Tests for json_repair integration in generate_structured / agenerate_structured."""

from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from api.core.shared import llm as llm_module
from api.core.shared.llm import invoke_structured_completion


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
