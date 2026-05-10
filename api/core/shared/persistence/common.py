from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def epoch_to_utc(value: float | datetime | None, *, default: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if value is None:
        return ensure_utc(default or utc_now())
    return datetime.fromtimestamp(float(value), tz=UTC)


def optional_epoch_to_utc(value: float | datetime | None) -> datetime | None:
    if value is None:
        return None
    return epoch_to_utc(value)


def utc_to_epoch(value: datetime | float | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, datetime):
        return ensure_utc(value).timestamp()
    return float(value)


def _is_numpy_value(value: Any) -> bool:
    return type(value).__module__.startswith("numpy")


def normalize_for_bson(value: Any) -> Any:  # noqa: C901, PLR0911
    if value is None or isinstance(value, str | int | float | bool | datetime):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, BaseModel):
        return normalize_for_bson(value.model_dump(by_alias=True))
    if is_dataclass(value):
        return normalize_for_bson(asdict(value))
    if _is_numpy_value(value):
        if hasattr(value, "tolist"):
            return normalize_for_bson(value.tolist())
        if hasattr(value, "item"):
            return normalize_for_bson(value.item())
    if isinstance(value, dict):
        return {str(key): normalize_for_bson(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [normalize_for_bson(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return normalize_for_bson(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return normalize_for_bson(value.dict())
    return value
