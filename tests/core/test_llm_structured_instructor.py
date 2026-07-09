"""Tests for the instructor-backed structured output path in api.shared.llm.

generate_structured / agenerate_structured delegate schema injection +
validation reask to instructor (Mode.JSON), composing our tenacity policy via
``max_retries``. These tests mock litellm ``completion`` / ``acompletion`` to
return real ModelResponse objects so instructor can parse them.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from litellm import ModelResponse, Usage
from litellm.types.utils import Choices, Message
from pydantic import BaseModel
import pytest

from api.shared import llm as llm_module
from api.shared.llm import ainvoke_structured_completion


class ExerciseSchema(BaseModel):
    exercise_type: str
    question: str


FAKE_SETTINGS = SimpleNamespace(
    llm_base_url="https://api.example.com",
    llm_api_key="test-key",  # pragma: allowlist secret
    llm_model="model-x",
    llm_timeout_sec=30,
    exercise_llm_model=None,
    llm_custom_provider=None,
    llm_retry_attempts=1,
    llm_retry_backoff_sec=0.0,
)


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                index=0, message=Message(role="assistant", content=content), finish_reason="stop"
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _patch_acompletion(monkeypatch, sync_fn):
    """Patch litellm.acompletion with an async shim over a plain sync callable."""

    async def _acompletion(**kwargs):
        return sync_fn(**kwargs)

    monkeypatch.setattr(llm_module, "acompletion", _acompletion)
    monkeypatch.setattr(llm_module, "get_settings", lambda: FAKE_SETTINGS)


def _call():
    return asyncio.run(
        ainvoke_structured_completion(
            schema=ExerciseSchema,
            model="model-x",
            messages=[{"role": "user", "content": "generate"}],
        )
    )


def test_valid_json_parses_into_schema(monkeypatch):
    _patch_acompletion(
        monkeypatch,
        lambda **_: _response('{"exercise_type": "fill_blank", "question": "Q?"}'),
    )
    result = _call()
    assert result.model_dump() == {"exercise_type": "fill_blank", "question": "Q?"}


def test_reask_on_validation_error_then_succeeds(monkeypatch):
    """instructor appends the validation error to the messages and retries.

    First response is missing a required field (invalid); second is valid.
    The client's tenacity policy allows the reask, so the call ultimately
    returns the parsed model. This is the value instructor adds over a plain
    retry: the second attempt gets feedback about what was wrong.
    """
    monkeypatch.setattr(llm_module, "resolve_llm_retry_policy", lambda: (3, 0.0))
    calls = {"n": 0}

    def _flaky(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response('{"exercise_type": "mcq"}')  # missing `question`
        return _response('{"exercise_type": "mcq", "question": "Q?"}')

    _patch_acompletion(monkeypatch, _flaky)

    result = _call()

    assert calls["n"] == 2
    assert result.model_dump() == {"exercise_type": "mcq", "question": "Q?"}


def test_retries_transient_provider_failure_then_succeeds(monkeypatch):
    """A transient provider error (raised before any response) is retried by the
    same tenacity policy that drives reask."""
    monkeypatch.setattr(llm_module, "resolve_llm_retry_policy", lambda: (3, 0.0))
    calls = {"n": 0}

    def _flaky(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient provider failure")
        return _response('{"exercise_type": "mcq", "question": "Q?"}')

    _patch_acompletion(monkeypatch, _flaky)

    result = _call()

    assert calls["n"] == 2
    assert result.model_dump() == {"exercise_type": "mcq", "question": "Q?"}


def test_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm_module, "resolve_llm_retry_policy", lambda: (2, 0.0))
    calls = {"n": 0}

    def _always_fail(**_):
        calls["n"] += 1
        raise RuntimeError("permanent provider failure")

    _patch_acompletion(monkeypatch, _always_fail)

    with pytest.raises(Exception, match="permanent provider failure"):
        _call()
    assert calls["n"] == 2
