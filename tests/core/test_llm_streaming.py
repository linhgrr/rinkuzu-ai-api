from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from api.shared import llm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _FakeStream:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> dict[str, object]:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _chunk(content: str, finish_reason: str | None) -> dict[str, object]:
    return {
        "choices": [
            {
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


def _client(*, max_attempts: int = 1) -> llm.LiteLLMClient:
    return llm.LiteLLMClient(
        config=llm.LLMProviderConfig(
            base_url="https://example.invalid",
            api_key="test-key",  # pragma: allowlist secret
            model="test-model",
            timeout_sec=5,
        ),
        max_attempts=max_attempts,
        base_delay_sec=0,
    )


async def _stream(client: llm.LiteLLMClient) -> AsyncIterator[str]:
    async for delta in client.stream_text(
        messages=[{"role": "user", "content": "Explain"}],
        max_tokens=1024,
    ):
        yield delta


@pytest.mark.anyio
async def test_stream_text_finishes_normally_when_provider_reports_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**_kwargs: object) -> _FakeStream:
        return _FakeStream([_chunk("Complete answer.", "stop")])

    monkeypatch.setattr(llm, "acompletion", fake_acompletion)

    assert [delta async for delta in _stream(_client())] == ["Complete answer."]


@pytest.mark.anyio
async def test_stream_text_surfaces_provider_output_truncation_after_partial_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**_kwargs: object) -> _FakeStream:
        return _FakeStream(
            [
                _chunk("Partial ", None),
                _chunk("answer", "length"),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 1024,
                        "total_tokens": 1044,
                    },
                },
            ]
        )

    async def ignore_usage(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(llm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm, "record_llm_usage", ignore_usage)

    stream = _stream(_client())
    assert await anext(stream) == "Partial "
    assert await anext(stream) == "answer"
    with pytest.raises(llm.LLMOutputTruncatedError, match="length limit"):
        await anext(stream)


@pytest.mark.anyio
async def test_stream_text_surfaces_truncation_before_first_visible_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_acompletion(**_kwargs: object) -> _FakeStream:
        nonlocal call_count
        call_count += 1
        return _FakeStream([_chunk("", "length")])

    monkeypatch.setattr(llm, "acompletion", fake_acompletion)

    with pytest.raises(llm.LLMOutputTruncatedError, match="visible text"):
        await anext(_stream(_client(max_attempts=3)))

    assert call_count == 1
