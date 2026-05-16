from __future__ import annotations

import json
from pathlib import Path
import re

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parent
BACKEND_CONTRACT_PATH = BACKEND_ROOT / "contracts" / "api-error-codes.json"
FRONTEND_CONTRACT_PATH = WORKSPACE_ROOT / "rinkuzu" / "contracts" / "api-error-codes.json"
BACKEND_CODE_REGEX = re.compile(
    r"(?:^|[^\w])(?:code|error_code)\s*=\s*[\"']([a-z0-9_]+)[\"']",
    re.MULTILINE,
)
IGNORED_DIR_NAMES = {".git", ".venv", ".worktrees", "__pycache__"}


def _load_contract(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _collect_backend_codes() -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    api_root = BACKEND_ROOT / "api"
    for file_path in _iter_python_files(api_root):
        repo_path = file_path.relative_to(BACKEND_ROOT).as_posix()
        source = file_path.read_text(encoding="utf-8")
        for match in BACKEND_CODE_REGEX.finditer(source):
            code = match.group(1)
            results.setdefault(code, []).append(repo_path)
    return {code: sorted(set(paths)) for code, paths in results.items()}


def test_api_error_code_contract_is_shared_with_frontend() -> None:
    assert _load_contract(BACKEND_CONTRACT_PATH) == _load_contract(FRONTEND_CONTRACT_PATH)


def test_api_error_code_contract_is_internally_consistent() -> None:
    contract = _load_contract(BACKEND_CONTRACT_PATH)
    default_messages = contract["defaultMessagesByCode"]
    assert isinstance(default_messages, dict)

    for status, code in contract["httpStatusToCode"].items():
        assert code in default_messages, (
            f"httpStatusToCode[{status}] -> {code} missing from defaultMessagesByCode"
        )

    for status in contract["httpStatusToMessage"]:
        assert status in contract["httpStatusToCode"], (
            f"httpStatusToMessage[{status}] has no matching httpStatusToCode entry"
        )


def test_all_backend_api_error_codes_are_declared_in_contract() -> None:
    contract = _load_contract(BACKEND_CONTRACT_PATH)
    allowed_codes = set(contract["defaultMessagesByCode"])
    usages = _collect_backend_codes()
    missing = {code: paths for code, paths in sorted(usages.items()) if code not in allowed_codes}

    assert not missing, f"Missing backend API error codes in contract: {missing}"
