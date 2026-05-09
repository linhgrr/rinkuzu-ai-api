"""Official OpenAI Files + Responses client for content-pipeline extraction."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from loguru import logger
from openai import APIError, BadRequestError, NotFoundError, OpenAI
from pydantic import BaseModel
from pymongo import ASCENDING, MongoClient

from api.config import get_settings

if TYPE_CHECKING:
    from pymongo.collection import Collection

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
class FileCacheEntry:
    file_id: str
    purpose: str
    cached_at: float


@dataclass(frozen=True)
class UploadedFileRef:
    file_id: str
    purpose: str
    cache_hit: bool


class StructuredExtractionClient(Protocol):
    """Provider boundary for file-backed structured extraction."""

    def upload_pdf_bytes(
        self,
        *,
        filename: str,
        pdf_bytes: bytes,
        sha256: str,
        now_ts: float,
        job_id: str | None = None,
    ) -> UploadedFileRef: ...

    def invalidate_cached_file(self, *, sha256: str) -> None: ...

    def parse_response(
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
        request_timeout_sec=settings.content_pipeline_responses_timeout_sec,
        max_retries=max(1, settings.llm_max_retries),
    )


class MongoFileCache:
    """Minimal persistent cache for uploaded OpenAI files."""

    def __init__(self, mongodb_uri: str | None, ttl_hours: int):
        self._mongodb_uri = (mongodb_uri or "").strip()
        self._ttl_seconds = max(1, ttl_hours) * 3600
        self._client: MongoClient[Any] | None = None
        self._collection: Collection[Any] | None = None

    def _get_collection(self) -> Collection[Any] | None:
        if not self._mongodb_uri:
            return None
        if self._collection is not None:
            return self._collection
        self._client = MongoClient(self._mongodb_uri, serverSelectionTimeoutMS=3000)
        collection = self._client["adaptive_learning"]["al_openai_file_cache"]
        collection.create_index(
            [("provider_fingerprint", ASCENDING), ("sha256", ASCENDING)],
            unique=True,
        )
        collection.create_index("cached_at")
        self._collection = collection
        return collection

    def load(self, *, provider_fingerprint: str, sha256: str, now_ts: float) -> FileCacheEntry | None:
        collection = self._get_collection()
        if collection is None:
            return None
        doc = collection.find_one(
            {"provider_fingerprint": provider_fingerprint, "sha256": sha256},
            {"_id": 0, "file_id": 1, "purpose": 1, "cached_at": 1},
        )
        if not doc:
            return None
        cached_at = float(doc.get("cached_at", 0))
        if now_ts - cached_at > self._ttl_seconds:
            collection.delete_one({"provider_fingerprint": provider_fingerprint, "sha256": sha256})
            return None
        file_id = str(doc.get("file_id") or "").strip()
        purpose = str(doc.get("purpose") or "").strip()
        if not file_id or not purpose:
            return None
        return FileCacheEntry(file_id=file_id, purpose=purpose, cached_at=cached_at)

    def save(
        self,
        *,
        provider_fingerprint: str,
        sha256: str,
        file_id: str,
        purpose: str,
        cached_at: float,
    ) -> None:
        collection = self._get_collection()
        if collection is None:
            return
        collection.update_one(
            {"provider_fingerprint": provider_fingerprint, "sha256": sha256},
            {
                "$set": {
                    "provider_fingerprint": provider_fingerprint,
                    "sha256": sha256,
                    "file_id": file_id,
                    "purpose": purpose,
                    "cached_at": cached_at,
                }
            },
            upsert=True,
        )

    def delete(self, *, provider_fingerprint: str, sha256: str) -> None:
        collection = self._get_collection()
        if collection is None:
            return
        collection.delete_one({"provider_fingerprint": provider_fingerprint, "sha256": sha256})


class OpenAIResponsesClient:
    """Thin wrapper around the official OpenAI Files + Responses APIs."""

    def __init__(self, *, config: ProviderConfig | None = None, cache: MongoFileCache | None = None, client: Any | None = None):
        settings = get_settings()
        self.config = config or build_provider_config()
        self.cache = cache or MongoFileCache(
            settings.mongodb_uri,
            settings.content_pipeline_file_cache_ttl_hours,
        )
        self._client = client or OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout_sec,
            max_retries=self.config.max_retries,
        )

    def upload_pdf_bytes(
        self,
        *,
        filename: str,
        pdf_bytes: bytes,
        sha256: str,
        now_ts: float,
        job_id: str | None = None,
    ) -> UploadedFileRef:
        cached = self.cache.load(
            provider_fingerprint=self.config.fingerprint,
            sha256=sha256,
            now_ts=now_ts,
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
            payload = self._client.files.create(
                file=(filename, io.BytesIO(pdf_bytes), "application/pdf"),
                purpose=_FILE_PURPOSE,
            )
        except APIError as exc:
            message = _api_error_message(exc)
            if _looks_like_payload_too_large(message):
                raise PayloadTooLargeError(message) from exc
            raise RuntimeError(message) from exc

        file_id = str(getattr(payload, "id", "") or "").strip()
        if not file_id:
            raise RuntimeError(f"OpenAI did not return a file id: {payload}")

        self.cache.save(
            provider_fingerprint=self.config.fingerprint,
            sha256=sha256,
            file_id=file_id,
            purpose=_FILE_PURPOSE,
            cached_at=now_ts,
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

    def invalidate_cached_file(self, *, sha256: str) -> None:
        self.cache.delete(provider_fingerprint=self.config.fingerprint, sha256=sha256)

    def parse_response(
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
            response = self._client.responses.parse(
                model=self.config.model,
                instructions=instructions,
                input=[{"role": "user", "content": input_blocks}],
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
