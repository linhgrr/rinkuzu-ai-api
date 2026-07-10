from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.config import get_settings
from api.dependencies import get_current_admin_user
from api.exceptions import AppError
from api.schemas.admin_ocr_keys import (
    OcrProviderKeyCreateRequest,
    OcrProviderKeyDeleteResponse,
    OcrProviderKeyListResponse,
    OcrProviderKeyPatchRequest,
    OcrProviderKeySingleResponse,
    OcrProviderKeyTestResponse,
)
from api.schemas.common import StandardResponse, ok
from api.shared import mongo_store
from api.shared.document_text import OCRProviderRequestError, check_ocr_api_key
from api.shared.ocr_key_crypto import OcrKeyCryptoError, decrypt_ocr_key
from api.shared.persistence import (
    OcrProviderKeyDuplicateError,
    create_ocr_provider_key,
    delete_ocr_provider_key,
    list_ocr_provider_keys,
    load_ocr_provider_key,
    load_ocr_provider_key_secret,
    record_ocr_key_failure,
    record_ocr_key_success,
    update_ocr_provider_key,
)

router = APIRouter(prefix="/api/v1/admin/ocr-keys", tags=["admin-ocr-keys"])


def _require_mongo() -> None:
    if not mongo_store.is_available():
        raise AppError(
            code="service_unavailable",
            message="Service unavailable",
            detail="MongoDB is required for OCR key management.",
            status_code=503,
        )


def _configuration_error(exc: Exception) -> AppError:
    return AppError(
        code="configuration_error",
        message="Configuration error",
        detail=str(exc),
        status_code=500,
    )


def _duplicate_error(exc: Exception) -> AppError:
    return AppError(
        code="conflict",
        message="OCR key already exists",
        detail=str(exc),
        status_code=409,
    )


async def _load_public_key_or_404(key_id: str) -> dict[str, object]:
    key = await load_ocr_provider_key(key_id=key_id)
    if key is None:
        raise AppError(
            code="not_found",
            message="OCR key not found",
            detail=f"OCR key {key_id} not found",
            status_code=404,
        )
    return key


@router.get("", response_model=StandardResponse[OcrProviderKeyListResponse])
async def list_admin_ocr_keys(
    _admin_user_id: Annotated[str, Depends(get_current_admin_user)],
) -> dict[str, object]:
    _require_mongo()
    keys = await list_ocr_provider_keys()
    return ok(
        {
            "keys": keys,
            "fallback_env_configured": bool((get_settings().ocr_api_key or "").strip()),
        }
    )


@router.post("", response_model=StandardResponse[OcrProviderKeySingleResponse], status_code=201)
async def create_admin_ocr_key(
    req: OcrProviderKeyCreateRequest,
    admin_user_id: Annotated[str, Depends(get_current_admin_user)],
) -> dict[str, object]:
    _require_mongo()
    try:
        key = await create_ocr_provider_key(
            label=req.label,
            api_key=req.api_key,
            priority=req.priority,
            actor_user_id=admin_user_id,
        )
    except OcrKeyCryptoError as exc:
        raise _configuration_error(exc) from exc
    except OcrProviderKeyDuplicateError as exc:
        raise _duplicate_error(exc) from exc
    return ok({"key": key})


@router.patch("/{key_id}", response_model=StandardResponse[OcrProviderKeySingleResponse])
async def update_admin_ocr_key(
    key_id: str,
    req: OcrProviderKeyPatchRequest,
    admin_user_id: Annotated[str, Depends(get_current_admin_user)],
) -> dict[str, object]:
    _require_mongo()
    try:
        key = await update_ocr_provider_key(
            key_id=key_id,
            label=req.label,
            api_key=req.api_key,
            priority=req.priority,
            enabled=req.enabled,
            actor_user_id=admin_user_id,
        )
    except OcrKeyCryptoError as exc:
        raise _configuration_error(exc) from exc
    except OcrProviderKeyDuplicateError as exc:
        raise _duplicate_error(exc) from exc
    if key is None:
        raise AppError(
            code="not_found",
            message="OCR key not found",
            detail=f"OCR key {key_id} not found",
            status_code=404,
        )
    return ok({"key": key})


@router.post("/{key_id}/test", response_model=StandardResponse[OcrProviderKeyTestResponse])
async def run_admin_ocr_key_test(
    key_id: str,
    _admin_user_id: Annotated[str, Depends(get_current_admin_user)],
) -> dict[str, object]:
    _require_mongo()
    secret_record = await load_ocr_provider_key_secret(key_id=key_id)
    if secret_record is None:
        raise AppError(
            code="not_found",
            message="OCR key not found",
            detail=f"OCR key {key_id} not found",
            status_code=404,
        )

    try:
        api_key = decrypt_ocr_key(str(secret_record.get("encrypted_key") or ""))
        await check_ocr_api_key(api_key)
    except OcrKeyCryptoError as exc:
        await record_ocr_key_failure(
            key_id=key_id,
            error_code="ocr_key_decrypt_failed",
            error_message=str(exc),
            disable=True,
        )
        key = await _load_public_key_or_404(key_id)
        return ok(
            {
                "key": key,
                "ok": False,
                "error_code": "ocr_key_decrypt_failed",
                "error_message": str(exc),
            }
        )
    except OCRProviderRequestError as exc:
        await record_ocr_key_failure(
            key_id=key_id,
            error_code=exc.error_code,
            error_message=str(exc),
            disable=exc.disable_key,
        )
        key = await _load_public_key_or_404(key_id)
        return ok(
            {
                "key": key,
                "ok": False,
                "error_code": exc.error_code,
                "error_message": str(exc),
            }
        )
    except Exception:
        await record_ocr_key_failure(
            key_id=key_id,
            error_code="ocr_request_failed",
            error_message="OCR key test failed.",
            disable=False,
        )
        key = await _load_public_key_or_404(key_id)
        return ok(
            {
                "key": key,
                "ok": False,
                "error_code": "ocr_request_failed",
                "error_message": "OCR key test failed.",
            }
        )

    await record_ocr_key_success(key_id=key_id)
    key = await _load_public_key_or_404(key_id)
    return ok({"key": key, "ok": True})


@router.delete("/{key_id}", response_model=StandardResponse[OcrProviderKeyDeleteResponse])
async def delete_admin_ocr_key(
    key_id: str,
    _admin_user_id: Annotated[str, Depends(get_current_admin_user)],
) -> dict[str, object]:
    _require_mongo()
    deleted = await delete_ocr_provider_key(key_id=key_id)
    if not deleted:
        raise AppError(
            code="not_found",
            message="OCR key not found",
            detail=f"OCR key {key_id} not found",
            status_code=404,
        )
    return ok({"deleted": True, "key_id": key_id})
