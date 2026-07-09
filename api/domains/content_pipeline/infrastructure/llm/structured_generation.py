"""Structured text-generation clients used by the content pipeline."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from api.config import get_settings
from api.shared.llm import (
    LiteLLMClient,
    LLMClient,
    LLMConfigurationError,
    LLMProviderConfig,
    build_llm_provider_config,
)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class ProviderConfigError(RuntimeError):
    """Raised when required provider settings are missing."""


class StructuredGenerationClient(Protocol):
    """Provider boundary for structured text generation."""

    async def parse_response(
        self,
        *,
        instructions: str,
        user_text: str,
        text_format: type[StructuredModelT],
        job_id: str | None = None,
        action: str | None = None,
    ) -> StructuredModelT:
        raise NotImplementedError


def build_provider_config() -> LLMProviderConfig:
    settings = get_settings()
    try:
        return build_llm_provider_config(
            model=settings.llm_model,
            timeout=settings.content_pipeline_llm_request_timeout_sec,
            max_retries=0,
        )
    except LLMConfigurationError as exc:
        raise ProviderConfigError(str(exc)) from exc


class LiteLLMStructuredGenerationClient:
    """Thin structured-output client over the shared LiteLLM adapter."""

    def __init__(
        self,
        *,
        config: LLMProviderConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or build_provider_config()
        self.llm_client = llm_client or LiteLLMClient(config=self.config)

    async def parse_response(
        self,
        *,
        instructions: str,
        user_text: str,
        text_format: type[StructuredModelT],
        job_id: str | None = None,
        action: str | None = None,
    ) -> StructuredModelT:
        del job_id
        return await self.llm_client.agenerate_structured(
            schema=text_format,
            temperature=0.0,
            action=action,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_text},
            ],
        )
