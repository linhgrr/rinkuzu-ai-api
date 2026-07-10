from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from api.config import get_settings

LONG_KEY_MASK_PREFIX_THRESHOLD = 12
MASKED_KEY_MIDDLE_LENGTH = 8


class OcrKeyCryptoError(RuntimeError):
    """Raised when OCR key encryption is not available or ciphertext is invalid."""


def _encryption_secret() -> str:
    secret = (get_settings().ocr_key_encryption_secret or "").strip()
    if not secret:
        raise OcrKeyCryptoError("OCR_KEY_ENCRYPTION_SECRET is not configured.")
    return secret


def _fernet_from_secret(secret: str) -> Fernet:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_ocr_key(raw_key: str) -> str:
    normalized = raw_key.strip()
    if not normalized:
        raise OcrKeyCryptoError("OCR key is empty.")
    return (
        _fernet_from_secret(_encryption_secret())
        .encrypt(normalized.encode("utf-8"))
        .decode("utf-8")
    )


def decrypt_ocr_key(encrypted_key: str) -> str:
    try:
        return (
            _fernet_from_secret(_encryption_secret())
            .decrypt(encrypted_key.encode("utf-8"))
            .decode("utf-8")
        )
    except InvalidToken as exc:
        raise OcrKeyCryptoError("OCR key cannot be decrypted.") from exc


def fingerprint_ocr_key(raw_key: str, *, provider: str = "landingai") -> str:
    normalized = raw_key.strip()
    return hashlib.sha256(f"{provider}:{normalized}".encode()).hexdigest()


def mask_ocr_key(raw_key: str) -> str:
    normalized = raw_key.strip()
    if not normalized:
        return ""
    last = normalized[-4:]
    prefix = normalized[:4] if len(normalized) > LONG_KEY_MASK_PREFIX_THRESHOLD else ""
    return f"{prefix}{'•' * MASKED_KEY_MIDDLE_LENGTH}{last}"
