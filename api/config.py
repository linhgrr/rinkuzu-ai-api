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
    adaptive_exercise_llm_model: Optional[str] = None
    llm_embedding_model: str = "text-embedding-3-small"
    llm_timeout_sec: float = 150
    llm_max_retries: int = 2
    adaptive_llm_max_workers: int = 8
    adaptive_llm_max_concurrency: Optional[int] = None
    adaptive_llm_timeout_sec: float = 120
    adaptive_prefetch_llm_timeout_sec: Optional[float] = None
    adaptive_llm_retry_attempts: int = 3
    adaptive_llm_retry_backoff_sec: float = 1.0

    # ── Content Pipeline ───────────────────────────────────
    embedding_model: str = "keepitreal/vietnamese-sbert"
    embedding_batch_size: int = 32
    use_vi_tokenizer: bool = False
    max_seq_length: Optional[int] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    prs_threshold: float = 0.75
    adaptive_mastery_threshold: float = 0.75
    similarity_threshold: float = 0.9
    pdf_ocr_concurrency: int = 5
    vision_pdf_request_timeout_sec: float = 120
    vision_agent_api_key: Optional[str] = None
    content_pipeline_job_timeout_sec: float = 1800
    content_pipeline_stage_timeout_sec: float = 300
    content_pipeline_graph_cycle_timeout_sec: float = 900

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


# Module-level alias for code importing `api.config.settings` directly.
settings = get_settings()
