"""
config.py — Centralized application settings using Pydantic BaseSettings.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings — auto-loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Models ──────────────────────────────────────────────
    load_models: bool = True
    saint_path: str = str(BASE_DIR / "models" / "saint_best.pt")
    dqn_path: str = str(BASE_DIR / "models" / "dqn_best.pt")

    # ── App Config ──────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    # ── MongoDB ─────────────────────────────────────────────
    mongo_url: Optional[str] = None

    # ── LLM ─────────────────────────────────────────────────
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None

    # ── Content Pipeline ───────────────────────────────────
    embedding_model: str = "keepitreal/vietnamese-sbert"
    embedding_batch_size: int = 32

    # ── S3 Cache ────────────────────────────────────────────
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_bucket_name: Optional[str] = None

    # ── Google/Gemini (legacy) ──────────────────────────────
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    @property
    def s3_available(self) -> bool:
        return all([self.s3_endpoint_url, self.s3_access_key_id, self.s3_secret_access_key])


# Singleton — imported by dependencies.py
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Module-level alias so `from config import settings` works
# (content-processor modules expect this name)
settings = get_settings()
