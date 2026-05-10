"""Official OpenAI Files + Responses client for content-pipeline extraction."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any, Protocol, TypeVar

from loguru import logger
from openai import APIError, AsyncOpenAI, BadRequestError, NotFoundError
from pydantic import BaseModel

from api.config import get_settings
from api.core.shared.persistence import (
    delete_cached_openai_file,
    load_cached_openai_file,
    save_cached_openai_file,
)

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_FILE_PURPOSE = "user_data"

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class FileReferenceError(RuntimeError):
    """Raised when OpenAI rejects a cached file reference."""


class PayloadTooLargeError(RuntimeError):
    """Raised when OpenAI rejects the request body size."""


class ProviderConfigError(RuntimeError):
    """Raised when required OpenAI settings are missing."""


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    model: str
    fingerprint: str
    request_timeout_sec: float
    max_retries: int


@dataclass(frozen=True)
class UploadedFileRef:
    file_id: str
    purpose: str
    cache_hit: bool


class StructuredExtractionClient(Protocol):
    """Provider boundary for file-backed structured extraction."""

    async def upload_pdf_bytes(
        self,
        *,
        filename: str,
        pdf_bytes: bytes,
        sha256: str,
        now_ts: float,
        job_id: str | None = None,
    ) -> UploadedFileRef: ...

    async def invalidate_cached_file(self, *, sha256: str) -> None: ...

    async def parse_response(
        self,
        *,
        instructions: str,
        input_blocks: list[dict[str, Any]],
        text_format: type[StructuredModelT],
        job_id: str | None = None,
    ) -> Any: ...


def normalize_openai_base_url(url: str | None) -> str:
    raw = (url or "").strip().rstrip("/")
    if not raw:
        return _DEFAULT_OPENAI_BASE_URL
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def build_provider_config() -> ProviderConfig:
    settings = get_settings()
    api_key = (settings.openai_api_key or "").strip()
    model = (settings.openai_model or "").strip()
    if not api_key:
        raise ProviderConfigError("OPENAI_API_KEY is required for the content pipeline.")
    if not model:
        raise ProviderConfigError("OPENAI_MODEL is required for the content pipeline.")

    base_url = normalize_openai_base_url(settings.openai_base_url)
    return ProviderConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        fingerprint=base_url,
        request_timeout_sec=settings.content_pipeline_llm_request_timeout_sec,
        max_retries=0,
    )


class OpenAIResponsesClient:
    """Thin wrapper around the official OpenAI Files + Responses APIs (async)."""

    def __init__(
        self,
        *,
        config: ProviderConfig | None = None,
        client: Any | None = None,
    ):
        self.config = config or build_provider_config()
        self._client = client or AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout_sec,
            max_retries=self.config.max_retries,
        )

    async def upload_pdf_bytes(
        self,
        *,
        filename: str,
        pdf_bytes: bytes,
        sha256: str,
        now_ts: float,
        job_id: str | None = None,
    ) -> UploadedFileRef:
        del now_ts
        cached = await load_cached_openai_file(
            provider_fingerprint=self.config.fingerprint,
            sha256=sha256,
        )
        if cached is not None:
            logger.info(
                "openai file cache hit job_id={} file_id={} purpose={} sha256={}",
                job_id or "-",
                cached.file_id,
                cached.purpose,
                sha256[:12],
            )
            return UploadedFileRef(file_id=cached.file_id, purpose=cached.purpose, cache_hit=True)

        logger.debug(
            "openai upload start job_id={} filename={} size_bytes={} purpose={} sha256={} base_url={}",
            job_id or "-",
            filename,
            len(pdf_bytes),
            _FILE_PURPOSE,
            sha256[:12],
            self.config.base_url,
        )
        try:
            payload = await self._client.files.create(
                file=(filename, io.BytesIO(pdf_bytes), "application/pdf"),
                purpose=_FILE_PURPOSE,  # type: ignore[arg-type]
            )
        except APIError as exc:
            message = _api_error_message(exc)
            if _looks_like_payload_too_large(message):
                raise PayloadTooLargeError(message) from exc
            raise RuntimeError(message) from exc

        file_id = str(getattr(payload, "id", "") or "").strip()
        if not file_id:
            raise RuntimeError(f"OpenAI did not return a file id: {payload}")

        await save_cached_openai_file(
            provider_fingerprint=self.config.fingerprint,
            sha256=sha256,
            file_id=file_id,
            purpose=_FILE_PURPOSE,
        )
        logger.debug(
            "openai upload done job_id={} filename={} file_id={} purpose={} size_bytes={}",
            job_id or "-",
            filename,
            file_id,
            _FILE_PURPOSE,
            len(pdf_bytes),
        )
        return UploadedFileRef(file_id=file_id, purpose=_FILE_PURPOSE, cache_hit=False)

    async def invalidate_cached_file(self, *, sha256: str) -> None:
        await delete_cached_openai_file(
            provider_fingerprint=self.config.fingerprint,
            sha256=sha256,
        )

    async def parse_response(
        self,
        *,
        instructions: str,
        input_blocks: list[dict[str, Any]],
        text_format: type[StructuredModelT],
        job_id: str | None = None,
    ) -> Any:
        logger.debug(
            "openai response start job_id={} model={} input_blocks={} schema={} base_url={}",
            job_id or "-",
            self.config.model,
            len(input_blocks),
            text_format.__name__,
            self.config.base_url,
        )
        try:
            response = await self._client.responses.parse(
                model=self.config.model,
                instructions=instructions,
                input=[{"role": "user", "content": input_blocks}],  # type: ignore[list-item,misc]
                text_format=text_format,
                store=False,
            )
        except (BadRequestError, NotFoundError) as exc:
            message = _api_error_message(exc)
            if _looks_like_missing_file(message):
                raise FileReferenceError(message) from exc
            if _looks_like_payload_too_large(message):
                raise PayloadTooLargeError(message) from exc
            raise RuntimeError(message) from exc
        except APIError as exc:
            message = _api_error_message(exc)
            if _looks_like_payload_too_large(message):
                raise PayloadTooLargeError(message) from exc
            raise RuntimeError(message) from exc

        logger.debug(
            "openai response done job_id={} model={} usage={} parsed={}",
            job_id or "-",
            self.config.model,
            response_usage_summary(response),
            type(getattr(response, "output_parsed", None)).__name__
            if getattr(response, "output_parsed", None) is not None
            else "none",
        )
        return response


def response_usage_summary(payload: Any) -> dict[str, int]:
    usage = payload.get("usage") if isinstance(payload, dict) else getattr(payload, "usage", None)
    if usage is None:
        return {}
    summary: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(value, int):
            summary[key] = value
    return summary


def require_parsed_output(response: Any, expected_type: type[StructuredModelT]) -> StructuredModelT:
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, expected_type):
        return parsed
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if output_text:
        raise RuntimeError(f"OpenAI did not return valid structured output: {output_text[:300]}")
    raise RuntimeError(
        f"OpenAI did not return valid structured output for schema {expected_type.__name__}."
    )


def _api_error_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return str(exc)


def _looks_like_missing_file(message: str) -> bool:
    lower = message.lower()
    return "file" in lower and ("not found" in lower or "invalid" in lower)


def _looks_like_payload_too_large(message: str) -> bool:
    lower = message.lower()
    return (
        "payload too large" in lower
        or "request entity too large" in lower
        or "function_payload_too_large" in lower
        or "content too large" in lower
    )
