"""Shared pytest bootstrap.

Hermetic defaults: test collection must never depend on a developer's local
``.env``. Set a deterministic, non-placeholder INTERNAL_SERVICE_TOKEN and dev
environment *before* any ``api`` import so module-level ``get_settings()`` is
safe. Do not print the token.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Deterministic >=32 token that is not a placeholder / repeated / all-same value.
# Set before api imports; pydantic-settings prefers env vars over env_file.
os.environ["ENVIRONMENT"] = "dev"
# Always override ambient env / local .env so collection is hermetic.
os.environ["INTERNAL_SERVICE_TOKEN"] = "test-internal-service-token-xx01"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
