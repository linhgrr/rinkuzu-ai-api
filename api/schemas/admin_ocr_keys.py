from __future__ import annotations

from datetime import (
    datetime,  # noqa: TC003 - Pydantic resolves this response annotation at runtime.
)
from typing import Literal

from pydantic import Field, model_validator

from api.schemas.common import BaseStandardModel

OcrKeyHealthStatus = Literal["untested", "healthy", "failed", "disabled"]


class OcrProviderKeyResponse(BaseStandardModel):
    key_id: str
    provider: str
    label: str
    masked_key: str
    enabled: bool
    health_status: OcrKeyHealthStatus
    priority: int
    success_count: int
    failure_count: int
    last_used_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class OcrProviderKeyListResponse(BaseStandardModel):
    keys: list[OcrProviderKeyResponse]
    fallback_env_configured: bool


class OcrProviderKeySingleResponse(BaseStandardModel):
    key: OcrProviderKeyResponse


class OcrProviderKeyTestResponse(BaseStandardModel):
    key: OcrProviderKeyResponse
    ok: bool
    error_code: str | None = None
    error_message: str | None = None


class OcrProviderKeyDeleteResponse(BaseStandardModel):
    deleted: bool
    key_id: str


class OcrProviderKeyCreateRequest(BaseStandardModel):
    label: str = Field(min_length=1, max_length=120)
    api_key: str = Field(min_length=1, max_length=500)
    priority: int = Field(default=100, ge=0, le=10_000)


class OcrProviderKeyPatchRequest(BaseStandardModel):
    label: str | None = Field(default=None, min_length=1, max_length=120)
    api_key: str | None = Field(default=None, min_length=1, max_length=500)
    priority: int | None = Field(default=None, ge=0, le=10_000)
    enabled: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> OcrProviderKeyPatchRequest:
        if (
            self.label is None
            and self.api_key is None
            and self.priority is None
            and self.enabled is None
        ):
            raise ValueError("At least one field must be provided.")
        return self
