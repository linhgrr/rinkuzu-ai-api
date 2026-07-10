from __future__ import annotations

from typing import Any
import uuid

from pymongo.errors import DuplicateKeyError

from api.shared.ocr_key_crypto import encrypt_ocr_key, fingerprint_ocr_key, mask_ocr_key

from .common import utc_now
from .documents import OcrProviderKeyDocument


class OcrProviderKeyDuplicateError(RuntimeError):
    """Raised when an OCR provider key already exists."""


def _public_key_doc(doc: OcrProviderKeyDocument) -> dict[str, Any]:
    status = "disabled" if not doc.enabled else doc.health_status
    return {
        "key_id": doc.key_id,
        "provider": doc.provider,
        "label": doc.label,
        "masked_key": doc.masked_key,
        "enabled": doc.enabled,
        "health_status": status,
        "priority": doc.priority,
        "success_count": doc.success_count,
        "failure_count": doc.failure_count,
        "last_used_at": doc.last_used_at,
        "last_success_at": doc.last_success_at,
        "last_error_at": doc.last_error_at,
        "last_error_code": doc.last_error_code,
        "last_error_message": doc.last_error_message,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }


async def list_ocr_provider_keys(provider: str = "landingai") -> list[dict[str, Any]]:
    docs = (
        await OcrProviderKeyDocument.find(OcrProviderKeyDocument.provider == provider)
        .sort("+priority", "-updated_at")
        .to_list()
    )
    return [_public_key_doc(doc) for doc in docs]


async def list_active_ocr_provider_key_secrets(
    provider: str = "landingai",
) -> list[dict[str, Any]]:
    docs = (
        await OcrProviderKeyDocument.find(
            OcrProviderKeyDocument.provider == provider,
            OcrProviderKeyDocument.enabled == True,  # noqa: E712
        )
        .sort("+priority", "+created_at")
        .to_list()
    )
    return [
        {
            "key_id": doc.key_id,
            "provider": doc.provider,
            "label": doc.label,
            "encrypted_key": doc.encrypted_key,
            "masked_key": doc.masked_key,
            "priority": doc.priority,
        }
        for doc in docs
    ]


async def load_ocr_provider_key(
    *,
    key_id: str,
    provider: str = "landingai",
) -> dict[str, Any] | None:
    doc = await OcrProviderKeyDocument.find_one(
        OcrProviderKeyDocument.provider == provider,
        OcrProviderKeyDocument.key_id == key_id,
    )
    return _public_key_doc(doc) if doc is not None else None


async def load_ocr_provider_key_secret(
    *,
    key_id: str,
    provider: str = "landingai",
) -> dict[str, Any] | None:
    doc = await OcrProviderKeyDocument.find_one(
        OcrProviderKeyDocument.provider == provider,
        OcrProviderKeyDocument.key_id == key_id,
    )
    if doc is None:
        return None
    return {
        "key_id": doc.key_id,
        "provider": doc.provider,
        "label": doc.label,
        "encrypted_key": doc.encrypted_key,
        "masked_key": doc.masked_key,
        "priority": doc.priority,
    }


async def create_ocr_provider_key(
    *,
    label: str,
    api_key: str,
    priority: int = 100,
    provider: str = "landingai",
    actor_user_id: str | None = None,
) -> dict[str, Any]:
    normalized_key = api_key.strip()
    doc = OcrProviderKeyDocument(
        key_id=uuid.uuid4().hex,
        provider=provider,
        label=label.strip(),
        encrypted_key=encrypt_ocr_key(normalized_key),
        key_fingerprint=fingerprint_ocr_key(normalized_key, provider=provider),
        masked_key=mask_ocr_key(normalized_key),
        priority=priority,
        created_by=actor_user_id,
        updated_by=actor_user_id,
    )
    try:
        await doc.insert()
    except DuplicateKeyError as exc:
        raise OcrProviderKeyDuplicateError("OCR key already exists.") from exc
    return _public_key_doc(doc)


async def update_ocr_provider_key(
    *,
    key_id: str,
    label: str | None = None,
    api_key: str | None = None,
    priority: int | None = None,
    enabled: bool | None = None,
    provider: str = "landingai",
    actor_user_id: str | None = None,
) -> dict[str, Any] | None:
    doc = await OcrProviderKeyDocument.find_one(
        OcrProviderKeyDocument.provider == provider,
        OcrProviderKeyDocument.key_id == key_id,
    )
    if doc is None:
        return None

    if label is not None:
        doc.label = label.strip()
    if priority is not None:
        doc.priority = priority
    if enabled is not None:
        doc.enabled = enabled
        if not enabled:
            doc.health_status = "disabled"
        elif doc.health_status == "disabled":
            doc.health_status = "untested"
    if api_key is not None:
        normalized_key = api_key.strip()
        doc.encrypted_key = encrypt_ocr_key(normalized_key)
        doc.key_fingerprint = fingerprint_ocr_key(normalized_key, provider=provider)
        doc.masked_key = mask_ocr_key(normalized_key)
        doc.health_status = "untested" if doc.enabled else "disabled"
        doc.success_count = 0
        doc.failure_count = 0
        doc.last_used_at = None
        doc.last_success_at = None
        doc.last_error_at = None
        doc.last_error_code = None
        doc.last_error_message = None
    doc.updated_by = actor_user_id

    try:
        await doc.replace()
    except DuplicateKeyError as exc:
        raise OcrProviderKeyDuplicateError("OCR key already exists.") from exc
    return _public_key_doc(doc)


async def delete_ocr_provider_key(*, key_id: str, provider: str = "landingai") -> bool:
    doc = await OcrProviderKeyDocument.find_one(
        OcrProviderKeyDocument.provider == provider,
        OcrProviderKeyDocument.key_id == key_id,
    )
    if doc is None:
        return False
    await doc.delete()
    return True


async def record_ocr_key_success(*, key_id: str, provider: str = "landingai") -> None:
    now = utc_now()
    doc = await OcrProviderKeyDocument.find_one(
        OcrProviderKeyDocument.provider == provider,
        OcrProviderKeyDocument.key_id == key_id,
    )
    if doc is None:
        return
    await doc.update(
        {
            "$set": {
                "enabled": True,
                "health_status": "healthy",
                "last_used_at": now,
                "last_success_at": now,
                "last_error_code": None,
                "last_error_message": None,
                "updated_at": now,
            },
            "$inc": {"success_count": 1},
        }
    )


async def record_ocr_key_failure(
    *,
    key_id: str,
    error_code: str,
    error_message: str,
    disable: bool,
    provider: str = "landingai",
) -> None:
    now = utc_now()
    payload: dict[str, Any] = {
        "health_status": "failed",
        "last_used_at": now,
        "last_error_at": now,
        "last_error_code": error_code[:80],
        "last_error_message": error_message[:300],
        "updated_at": now,
    }
    if disable:
        payload["enabled"] = False
    doc = await OcrProviderKeyDocument.find_one(
        OcrProviderKeyDocument.provider == provider,
        OcrProviderKeyDocument.key_id == key_id,
    )
    if doc is None:
        return
    await doc.update({"$set": payload, "$inc": {"failure_count": 1}})
