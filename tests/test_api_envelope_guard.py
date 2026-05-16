from __future__ import annotations

from pathlib import Path
import re

BACKEND_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = BACKEND_ROOT / "api"
IGNORED_DIR_NAMES = {".git", ".venv", ".worktrees", "__pycache__"}
ALLOWED_JSON_RESPONSE_FILES = {"api/exceptions.py"}
ALLOWED_RAW_SUCCESS_FILES = {"api/schemas/common.py"}
ALLOWED_RAW_ERROR_FILES = {"api/exceptions.py"}
RAW_SUCCESS_REGEX = re.compile(r"[\{\[]\s*[\"']success[\"']\s*:\s*True\b")
RAW_ERROR_REGEX = re.compile(r"[\{\[]\s*[\"']success[\"']\s*:\s*False\b")


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in API_ROOT.rglob("*.py"):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def test_jsonresponse_is_only_built_in_exception_boundary() -> None:
    offenders: list[str] = []
    for path in _iter_python_files():
        repo_path = path.relative_to(BACKEND_ROOT).as_posix()
        if repo_path in ALLOWED_JSON_RESPONSE_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        if "JSONResponse(" in source:
            offenders.append(repo_path)

    assert not offenders, f"Use ok()/AppError instead of JSONResponse directly: {offenders}"


def test_raw_success_envelopes_only_live_in_common_schema_helper() -> None:
    offenders: list[str] = []
    for path in _iter_python_files():
        repo_path = path.relative_to(BACKEND_ROOT).as_posix()
        if repo_path in ALLOWED_RAW_SUCCESS_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        if RAW_SUCCESS_REGEX.search(source):
            offenders.append(repo_path)

    assert not offenders, (
        f"Raw success envelopes must be built via api.schemas.common.ok(): {offenders}"
    )


def test_raw_error_envelopes_only_live_in_exception_boundary() -> None:
    offenders: list[str] = []
    for path in _iter_python_files():
        repo_path = path.relative_to(BACKEND_ROOT).as_posix()
        if repo_path in ALLOWED_RAW_ERROR_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        if RAW_ERROR_REGEX.search(source):
            offenders.append(repo_path)

    assert not offenders, (
        f"Raw error envelopes must be built via api.exceptions.error_json_response(): {offenders}"
    )
