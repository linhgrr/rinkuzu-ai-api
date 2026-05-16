from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

DataT = TypeVar("DataT")


def ok(data: object, meta: dict[str, object] | None = None) -> dict[str, object]:
    """Construct the standard success response envelope."""
    payload: dict[str, object] = {"success": True, "data": data}
    if meta is not None:
        payload["meta"] = meta
    return payload


class BaseStandardModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class StandardResponse(BaseStandardModel, Generic[DataT]):
    success: bool = True
    data: DataT
    meta: dict[str, object] | None = None


class ErrorDetail(BaseStandardModel):
    code: str
    message: str
    detail: str | None = None
    meta: dict[str, object] | list[dict[str, object]] | None = None


class StandardErrorResponse(BaseStandardModel):
    success: bool = False
    error: ErrorDetail
