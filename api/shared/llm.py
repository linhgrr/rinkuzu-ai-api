"""
llm.py — Shared LiteLLM-backed abstractions and helpers for the API codebase.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from json_repair import loads as repair_json_loads
from litellm import acompletion, completion, get_supported_openai_params
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

from api.config import get_settings
from api.shared.llm_usage import extract_usage, record_llm_usage
from api.shared.retry import (
    async_retry,
    is_retryable_llm_error,
    resolve_llm_retry_policy,
    sync_retry,
)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

_JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
_DEEPSEEK_PROVIDER = "deepseek"


class LLMConfigurationError(ValueError):
    """Raised when required LLM settings are missing."""


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

    def generate_text(
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

    def generate_structured(
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


def _default_headers() -> dict[str, str]:
    return {"ngrok-skip-browser-warning": "true"}


def _normalize_message_content(content: object) -> str:  # noqa: C901
    if isinstance(content, str):
        return content
    if isinstance(content, list):
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
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
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
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n\n".join(part for part in parts if part.strip()).strip()
    return str(content).strip()


def _resolve_message_role(message: object) -> str:
    if isinstance(message, dict):
        role = message.get("role")
        return str(role) if role else "user"

    message_type = getattr(message, "type", None)
    if message_type == "system":
        return "system"
    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "assistant"
    if message_type == "tool":
        return "tool"
    return "user"


def normalize_chat_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            content = _normalize_message_content(message.get("content", ""))
            role = str(message.get("role") or "user")
            payload: dict[str, Any] = {"role": role, "content": content}
            if role == "tool" and message.get("tool_call_id"):
                payload["tool_call_id"] = message["tool_call_id"]
            normalized.append(payload)
            continue

        normalized.append(
            {
                "role": _resolve_message_role(message),
                "content": _normalize_message_content(getattr(message, "content", "")),
            }
        )
    return normalized


def _augment_messages_for_schema(
    messages: Sequence[object],
    *,
    schema: type[StructuredModelT],
) -> list[dict[str, Any]]:
    normalized = normalize_chat_messages(messages)
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    guidance = (
        "Return valid json only. Không dùng markdown code fence. "
        "JSON phải khớp chính xác schema sau:\n"
        f"{schema_json}\n"
        "Nếu thiếu dữ liệu, vẫn phải trả về JSON hợp lệ và dùng giá trị rỗng/null phù hợp schema."
    )

    if normalized and normalized[0]["role"] == "system":
        normalized[0] = {
            **normalized[0],
            "content": f"{normalized[0]['content'].rstrip()}\n\n{guidance}",
        }
        return normalized

    return [{"role": "system", "content": guidance}, *normalized]


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
        "messages": normalize_chat_messages(messages),
        "api_key": config.api_key,
        "base_url": config.base_url,
        "timeout": config.timeout_sec,
        "num_retries": config.max_retries,
        "extra_headers": _default_headers(),
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


def _supports_openai_param(config: LLMProviderConfig, param: str) -> bool:
    try:
        supported = get_supported_openai_params(
            model=config.model,
            custom_llm_provider=config.custom_llm_provider,
        )
    except Exception as exc:
        logger.debug(
            "[LLM] could not resolve supported params for model={} provider={}: {}",
            config.model,
            config.custom_llm_provider or "(auto)",
            exc,
        )
        return False
    return param in (supported or [])


def _structured_response_format(config: LLMProviderConfig) -> dict[str, Any] | None:
    if not _supports_openai_param(config, "response_format"):
        return None
    return _JSON_OBJECT_RESPONSE_FORMAT


# Hold strong references to fire-and-forget tasks so they are not garbage
# collected mid-flight (see RUF006); each task removes itself when done.
_usage_tasks: set[asyncio.Task[None]] = set()


def _record_usage_sync(config: LLMProviderConfig, response: object, action: str | None) -> None:
    """Fire-and-forget usage recording from a sync context. Best-effort."""
    usage = extract_usage(response)
    if not usage:
        return
    coro = record_llm_usage(
        model=config.model,
        provider=config.custom_llm_provider,
        usage=usage,
        action=action,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (pure sync call) — run to completion best-effort.
        try:
            asyncio.run(coro)
        except Exception as exc:
            logger.debug("[llm_usage] sync record skipped: {}", exc)
            coro.close()
        return
    task = loop.create_task(coro)
    _usage_tasks.add(task)
    task.add_done_callback(_usage_tasks.discard)


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

    def _sync_retry(self, label: str) -> Any:
        return sync_retry(
            label=label,
            max_attempts=self._max_attempts,
            base_delay_sec=self._base_delay_sec,
            retry_on=is_retryable_llm_error,
        )

    def _async_retry(self, label: str) -> Any:
        return async_retry(
            label=label,
            max_attempts=self._max_attempts,
            base_delay_sec=self._base_delay_sec,
            retry_on=is_retryable_llm_error,
        )

    def generate_text(
        self,
        *,
        messages: Sequence[object],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> str:
        @self._sync_retry("generate_text")
        def _call() -> str:
            response = completion(
                **_litellm_kwargs(
                    config=self.config,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_enabled=thinking_enabled,
                )
            )
            _record_usage_sync(self.config, response, action)
            return extract_llm_text(_extract_response_content(response))

        return cast("str", _call())

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

        async def _open_to_first_token() -> tuple[str, Any]:
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
                if text:
                    return text, stream
            raise RuntimeError("LLM returned an empty completion")

        first_delta, stream = await self._async_retry("stream_text")(_open_to_first_token)()
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

    def generate_structured(
        self,
        *,
        messages: Sequence[object],
        schema: type[StructuredModelT],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        thinking_enabled: bool = False,
        action: str | None = None,
    ) -> StructuredModelT:
        @self._sync_retry("generate_structured")
        def _call() -> StructuredModelT:
            response = completion(
                **_litellm_kwargs(
                    config=self.config,
                    messages=_augment_messages_for_schema(messages, schema=schema),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_enabled=thinking_enabled,
                    response_format=_structured_response_format(self.config),
                )
            )
            _record_usage_sync(self.config, response, action)
            content = extract_llm_text(_extract_response_content(response))
            if not content:
                raise TypeError("LLM returned empty structured output.")
            return schema.model_validate(repair_json_loads(content))

        return cast("StructuredModelT", _call())

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
        @self._async_retry("agenerate_structured")
        async def _call() -> StructuredModelT:
            response = await acompletion(
                **_litellm_kwargs(
                    config=self.config,
                    messages=_augment_messages_for_schema(messages, schema=schema),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_enabled=thinking_enabled,
                    response_format=_structured_response_format(self.config),
                )
            )
            await _record_usage_async(self.config, response, action)
            content = extract_llm_text(_extract_response_content(response))
            if not content:
                raise TypeError("LLM returned empty structured output.")
            return schema.model_validate(repair_json_loads(content))

        return cast("StructuredModelT", await _call())


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


def invoke_text_completion(
    *,
    messages: Sequence[object],
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,
    max_tokens: int | None = None,
    thinking_enabled: bool = False,
    action: str | None = None,
) -> str:
    client = get_default_llm_client(model=model, timeout=timeout)
    return client.generate_text(
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


def invoke_structured_completion(
    *,
    messages: Sequence[object],
    schema: type[StructuredModelT],
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,
    max_tokens: int | None = None,
    thinking_enabled: bool = False,
    action: str | None = None,
) -> StructuredModelT:
    client = get_default_llm_client(model=model, timeout=timeout)
    return client.generate_structured(
        messages=messages,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_enabled=thinking_enabled,
        action=action,
    )


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
