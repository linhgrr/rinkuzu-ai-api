"""Compatibility shim for legacy pipeline modules.

This keeps top-level imports like ``from config import settings`` working
while the legacy content pipeline packages are being folded into the main
backend.
"""

from api.config import Settings, get_settings


settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]
