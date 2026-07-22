"""
llm.py — Shared LiteLLM-backed abstractions and helpers for the API codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import instructor
from litellm import acompletion, supports_vision
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

from api.config import get_settings
from api.shared.llm_usage import extract_usage, record_llm_usage
from api.shared.retry import (
    NonRetryableLLMError,
    async_retry,
    build_async_retrying,
    is_retryable_llm_error,
    resolve_llm_retry_policy,
)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

_DEEPSEEK_PROVIDER = "deepseek"
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


class LLMConfigurationError(ValueError):
    """Raised when required LLM settings are missing."""


class LLMOutputTruncatedError(NonRetryableLLMError):
    """Raised when the provider stops because the output-token limit was reached."""


@dataclass(frozen=True)
class LLMProviderConfig:
    """Normalized provider settings used by all LLM entry points."""

    base_url: str
    api_key: str
    model: str
    timeout_sec: float
    max_retries: int = 0
    custom_llm_provider: str | None = None


class LLMClient(Protocol):
    """Unified project-wide LLM capability surface."""

    async def agenerate_text(
        self,
        *,
        messages: Sequence[object],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> str:
        raise NotImplementedError

    def stream_text(
        self,
        *,
        messages: Sequence[object],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def agenerate_structured(
        self,
        *,
        messages: Sequence[object],
        schema: type[StructuredModelT],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> StructuredModelT:
        raise NotImplementedError


def normalize_llm_base_url(url: str | None) -> str:
    raw = (url or "").strip().rstrip("/")
    if not raw:
        raise LLMConfigurationError("LLM base URL is not set. Configure LLM_BASE_URL.")
    return raw


def _require_llm_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise LLMConfigurationError("LLM model is empty. Configure LLM_MODEL.")
    return normalized


def resolve_llm_api_key() -> str:
    settings = get_settings()
    key = cast("str | None", getattr(settings, "llm_api_key", None))
    if not key:
        raise LLMConfigurationError("LLM API key is not set. Configure LLM_API_KEY.")
    return key


def _resolve_default_llm_model(explicit_model: str | None = None) -> str:
    settings = get_settings()
    model = explicit_model or cast("str | None", getattr(settings, "llm_model", None))
    if not model:
        raise LLMConfigurationError("LLM model is not set. Configure LLM_MODEL.")
    return _require_llm_model(model)


def _resolve_custom_llm_provider(
    *,
    model: str,
    base_url: str,
    explicit_provider: str | None = None,
) -> str | None:
    provider = (explicit_provider or "").strip()
    if provider:
        return provider
    if model.strip().lower().startswith(f"{_DEEPSEEK_PROVIDER}/"):
        return _DEEPSEEK_PROVIDER
    if "api.deepseek.com" in base_url.lower():
        return _DEEPSEEK_PROVIDER
    return None


def build_llm_provider_config(
    *,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMProviderConfig:
    settings = get_settings()
    resolved_base_url = normalize_llm_base_url(
        base_url or cast("str | None", getattr(settings, "llm_base_url", None))
    )
    resolved_model = _resolve_default_llm_model(model)
    return LLMProviderConfig(
        base_url=resolved_base_url,
        api_key=api_key or resolve_llm_api_key(),
        model=resolved_model,
        timeout_sec=float(timeout or settings.llm_timeout_sec),
        max_retries=max_retries,
        custom_llm_provider=_resolve_custom_llm_provider(
            model=resolved_model,
            base_url=resolved_base_url,
            explicit_provider=cast(
                "str | None",
                getattr(settings, "llm_custom_provider", None),
            ),
        ),
    )


def _image_url_payload(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url.strip()}}


def _normalize_message_content(content: object, *, preserve_multimodal: bool = False) -> object:  # noqa: C901
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if preserve_multimodal:
            blocks: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        blocks.append({"type": "text", "text": item.strip()})
                    continue
                if not isinstance(item, dict):
                    text = str(item).strip()
                    if text:
                        blocks.append({"type": "text", "text": text})
                    continue

                item_type = str(item.get("type", "")).lower()
                if item_type in {"text", "input_text"}:
                    raw_text = item.get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        blocks.append({"type": "text", "text": raw_text.strip()})
                    continue

                if item_type in {"image", "image_url", "input_image"}:
                    image_value: object = item.get("url") or item.get("file_url")
                    if not image_value:
                        nested_image = item.get("image_url")
                        if isinstance(nested_image, dict):
                            image_value = nested_image.get("url")
                        else:
                            image_value = nested_image
                    if isinstance(image_value, str) and image_value.strip():
                        blocks.append(_image_url_payload(image_value))
                    continue

                raw_text = item.get("text")
                if isinstance(raw_text, str) and raw_text.strip():
                    blocks.append({"type": "text", "text": raw_text.strip()})
            return blocks

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = str(item.get("type", "")).lower()
            if item_type in {"text", "input_text"}:
                raw_text = item.get("text")
                if isinstance(raw_text, str) and raw_text.strip():
                    parts.append(raw_text.strip())
                continue
            if item_type in {"image", "image_url", "input_image"}:
                url = item.get("url") or item.get("image_url") or item.get("file_url")
                if isinstance(url, str) and url.strip():
                    parts.append(
                        "Hình ảnh tham khảo (model hiện tại chỉ đọc text, không đọc trực tiếp ảnh): "
                        + url.strip()
                    )
                continue
            if item_type in {"file", "input_file"}:
                filename = item.get("filename") or item.get("file_id") or item.get("file_url")
                if isinstance(filename, str) and filename.strip():
                    parts.append(
                        "Tệp đính kèm tham khảo (model hiện tại chỉ đọc text đã được trích xuất): "
                        + filename.strip()
                    )
                continue
            raw_text = item.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                parts.append(raw_text.strip())
        return "\n\n".join(part for part in parts if part.strip()).strip()
    return str(content).strip()


def _messages_contain_images(messages: Sequence[object]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and str(item.get("type", "")).lower() in {
                "image",
                "image_url",
                "input_image",
            }:
                return True
    return False


def normalize_chat_messages(
    messages: Sequence[object],
    *,
    model: str | None = None,
    custom_llm_provider: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize chat messages to litellm's ``{"role", "content"}`` dict shape.

    Text-only providers receive flattened text. Vision-capable models keep
    OpenAI-compatible ``image_url`` content blocks so LiteLLM can route them
    natively.
    """
    normalized: list[dict[str, Any]] = []
    preserve_multimodal = bool(
        model
        and _messages_contain_images(messages)
        and supports_vision(model=model, custom_llm_provider=custom_llm_provider)
    )
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"Expected a chat-message dict, got {type(message).__name__}.")
        content = _normalize_message_content(
            message.get("content", ""),
            preserve_multimodal=preserve_multimodal,
        )
        role = str(message.get("role") or "user")
        payload: dict[str, Any] = {"role": role, "content": content}
        if role == "tool" and message.get("tool_call_id"):
            payload["tool_call_id"] = message["tool_call_id"]
        normalized.append(payload)
    return normalized


def _extract_choice_content(choice: object) -> object:
    if isinstance(choice, dict):
        message = choice.get("message") or choice.get("delta") or {}
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    message = getattr(choice, "message", None) or getattr(choice, "delta", None)
    if message is None:
        return ""
    return getattr(message, "content", "")


def _extract_choice_finish_reason(choice: object) -> str | None:
    reason = (
        choice.get("finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    if reason is None:
        return None
    value = getattr(reason, "value", reason)
    return str(value).strip().lower() or None


def _raise_for_truncated_finish_reason(
    finish_reason: str | None, *, before_visible_text: bool
) -> None:
    if finish_reason not in _TRUNCATED_FINISH_REASONS:
        return
    detail = " before producing visible text" if before_visible_text else " before finishing"
    raise LLMOutputTruncatedError(
        f"Rin-chan reached the response length limit{detail}. Please ask to continue."
    )


def _extract_response_content(response: object) -> object:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices", [])
    if not choices:
        return ""
    return _extract_choice_content(choices[0])


def _extract_stream_delta_text(content: object) -> str:
    """Extract a streaming delta WITHOUT stripping whitespace.

    Unlike ``extract_llm_text`` (used for full-message extraction), inter-token
    spaces must be preserved during streaming, and a ``None``/missing delta must
    map to an empty string rather than the literal ``"None"``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _thinking_kwargs(thinking_enabled: bool) -> dict[str, Any]:  # noqa: FBT001
    if not thinking_enabled:
        return {}
    return {"thinking": {"type": "enabled"}}


def _litellm_kwargs(
    *,
    config: LLMProviderConfig,
    messages: Sequence[object],
    temperature: float,
    max_tokens: int | None,
    thinking_enabled: bool,
    response_format: dict[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": normalize_chat_messages(
            messages,
            model=config.model,
            custom_llm_provider=config.custom_llm_provider,
        ),
        "api_key": config.api_key,
        "base_url": config.base_url,
        "timeout": config.timeout_sec,
        "num_retries": config.max_retries,
        "temperature": temperature,
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
        **_thinking_kwargs(thinking_enabled),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    if config.custom_llm_provider:
        payload["custom_llm_provider"] = config.custom_llm_provider
    return payload


def _structured_litellm_kwargs(
    *,
    config: LLMProviderConfig,
    temperature: float,
    max_tokens: int | None,
    thinking_enabled: bool,
) -> dict[str, Any]:
    """litellm kwargs for instructor's structured path.

    instructor owns ``messages`` and ``response_model`` (JSON-mode schema
    injection + validation reask), so this passes only provider/model routing
    and generation params — no ``response_format``.
    """
    payload: dict[str, Any] = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "timeout": config.timeout_sec,
        "num_retries": config.max_retries,
        "temperature": temperature,
        **_thinking_kwargs(thinking_enabled),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if config.custom_llm_provider:
        payload["custom_llm_provider"] = config.custom_llm_provider
    return payload


async def _record_usage_async(
    config: LLMProviderConfig, response: object, action: str | None
) -> None:
    await record_llm_usage(
        model=config.model,
        provider=config.custom_llm_provider,
        usage=extract_usage(response),
        action=action,
    )


class LiteLLMClient(LLMClient):
    """Project-standard LLM client backed by LiteLLM.

    Retry is a client default: every LLM call retries transient failures
    (provider 5xx, timeouts, rate limits) using the ``llm_retry_*`` settings.
    Call sites no longer wrap calls in retry helpers — the client owns it.
    """

    def __init__(
        self,
        *,
        config: LLMProviderConfig,
        max_attempts: int | None = None,
        base_delay_sec: float | None = None,
    ) -> None:
        self.config = config
        policy_attempts, policy_delay = resolve_llm_retry_policy()
        self._max_attempts = policy_attempts if max_attempts is None else max_attempts
        self._base_delay_sec = policy_delay if base_delay_sec is None else base_delay_sec

    def _async_retry(self, label: str) -> Any:
        return async_retry(
            label=label,
            max_attempts=self._max_attempts,
            base_delay_sec=self._base_delay_sec,
            retry_on=is_retryable_llm_error,
        )

    async def agenerate_text(
        self,
        *,
        messages: Sequence[object],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> str:
        @self._async_retry("agenerate_text")
        async def _call() -> str:
            response = await acompletion(
                **_litellm_kwargs(
                    config=self.config,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_enabled=thinking_enabled,
                )
            )
            await _record_usage_async(self.config, response, action)
            return extract_llm_text(_extract_response_content(response))

        return cast("str", await _call())

    async def stream_text(
        self,
        *,
        messages: Sequence[object],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens, retrying only the open→first-token phase.

        A transient failure (or an empty completion) before the first token is
        retried transparently. Once the first token is yielded the stream is
        committed: a later failure surfaces to the caller mid-stream, because a
        partially-sent reply cannot be restarted.
        """

        async def _open_to_first_token() -> tuple[str, Any, str | None]:
            stream = await acompletion(
                **_litellm_kwargs(
                    config=self.config,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_enabled=thinking_enabled,
                    stream=True,
                )
            )
            async for chunk in stream:
                text = self._first_delta_from_chunk(chunk)
                finish_reason = self._finish_reason_from_chunk(chunk)
                if text:
                    return text, stream, finish_reason
                _raise_for_truncated_finish_reason(finish_reason, before_visible_text=True)
            raise RuntimeError("LLM returned an empty completion")

        first_delta, stream, finish_reason = await self._async_retry("stream_text")(
            _open_to_first_token
        )()
        yield first_delta

        final_usage: dict[str, int] | None = None
        async for chunk in stream:
            chunk_usage = extract_usage(chunk)
            if chunk_usage:
                final_usage = chunk_usage
            choices = getattr(chunk, "choices", None)
            if choices is None and isinstance(chunk, dict):
                choices = chunk.get("choices", [])
            for choice in choices or []:
                next_finish_reason = _extract_choice_finish_reason(choice)
                if next_finish_reason:
                    finish_reason = next_finish_reason
                text = _extract_stream_delta_text(_extract_choice_content(choice))
                if text:
                    yield text
        if final_usage:
            await record_llm_usage(
                model=self.config.model,
                provider=self.config.custom_llm_provider,
                usage=final_usage,
                action=action,
            )
        _raise_for_truncated_finish_reason(finish_reason, before_visible_text=False)

    @staticmethod
    def _first_delta_from_chunk(chunk: object) -> str:
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices", [])
        for choice in choices or []:
            text = _extract_stream_delta_text(_extract_choice_content(choice))
            if text:
                return text
        return ""

    @staticmethod
    def _finish_reason_from_chunk(chunk: object) -> str | None:
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices", [])
        for choice in choices or []:
            reason = _extract_choice_finish_reason(choice)
            if reason:
                return reason
        return None

    async def agenerate_structured(
        self,
        *,
        messages: Sequence[object],
        schema: type[StructuredModelT],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> StructuredModelT:
        client = cast(
            "instructor.AsyncInstructor",
            instructor.from_litellm(acompletion, mode=instructor.Mode.JSON),
        )
        result, raw = await client.chat.completions.create_with_completion(
            response_model=schema,
            messages=cast("Any", normalize_chat_messages(messages)),
            max_retries=cast(
                "Any",
                build_async_retrying(
                    label="agenerate_structured",
                    max_attempts=self._max_attempts,
                    base_delay_sec=self._base_delay_sec,
                    retry_on=is_retryable_llm_error,
                ),
            ),
            **_structured_litellm_kwargs(
                config=self.config,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_enabled=thinking_enabled,
            ),
        )
        await _record_usage_async(self.config, raw, action)
        return result


def get_default_llm_client(
    *,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
) -> LiteLLMClient:
    return LiteLLMClient(
        config=build_llm_provider_config(
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    )


async def ainvoke_text_completion(
    *,
    messages: Sequence[object],
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,  # noqa: ASYNC109
    max_tokens: int | None = None,
    thinking_enabled: bool = False,
    action: str | None = None,
) -> str:
    client = get_default_llm_client(model=model, timeout=timeout)
    return await client.agenerate_text(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_enabled=thinking_enabled,
        action=action,
    )


async def astream_text_completion(
    *,
    messages: Sequence[object],
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,  # noqa: ASYNC109
    max_tokens: int | None = None,
    thinking_enabled: bool = False,
    action: str | None = None,
) -> AsyncIterator[str]:
    client = get_default_llm_client(model=model, timeout=timeout)
    async for chunk in client.stream_text(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_enabled=thinking_enabled,
        action=action,
    ):
        yield chunk


async def ainvoke_structured_completion(
    *,
    messages: Sequence[object],
    schema: type[StructuredModelT],
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,  # noqa: ASYNC109
    max_tokens: int | None = None,
    thinking_enabled: bool = False,
    action: str | None = None,
) -> StructuredModelT:
    client = get_default_llm_client(model=model, timeout=timeout)
    return await client.agenerate_structured(
        messages=messages,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_enabled=thinking_enabled,
        action=action,
    )


SSE_STREAM_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def serialize_responses_sse_event(event: object) -> bytes:
    if hasattr(event, "model_dump_json"):
        payload = event.model_dump_json(exclude_none=True)
    elif hasattr(event, "to_json"):
        payload = event.to_json()
    else:
        payload = json.dumps(event, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


def _resolve_shared_llm_model(explicit_model: str | None) -> str:
    settings = get_settings()
    model = cast("str | None", getattr(settings, "exercise_llm_model", None))
    model = model or explicit_model or cast("str | None", getattr(settings, "llm_model", None))
    if not model:
        raise LLMConfigurationError(
            "LLM model is not set. Configure EXERCISE_LLM_MODEL or LLM_MODEL."
        )
    return _require_llm_model(model)


def extract_llm_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()
