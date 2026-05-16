from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast

_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "api-error-codes.json"
_CONTRACT = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))

DEFAULT_API_ERROR_MESSAGE_BY_CODE: Final[dict[str, str]] = cast(
    "dict[str, str]", _CONTRACT["defaultMessagesByCode"]
)
HTTP_ERROR_CODE_BY_STATUS: Final[dict[int, str]] = {
    int(status): str(code) for status, code in _CONTRACT["httpStatusToCode"].items()
}
HTTP_ERROR_MESSAGE_BY_STATUS: Final[dict[int, str]] = {
    int(status): str(message) for status, message in _CONTRACT["httpStatusToMessage"].items()
}


def get_default_api_error_message(code: str, fallback: str = "Request failed") -> str:
    return DEFAULT_API_ERROR_MESSAGE_BY_CODE.get(code, fallback)


def get_http_error_code(status_code: int, fallback: str = "internal_error") -> str:
    return HTTP_ERROR_CODE_BY_STATUS.get(status_code, fallback)


def get_http_error_message(status_code: int, fallback: str = "Internal server error") -> str:
    return HTTP_ERROR_MESSAGE_BY_STATUS.get(status_code, fallback)
