"""
config.py — Centralized application settings using Pydantic BaseSettings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
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
    environment: str = "dev"  # dev | staging | prod — controls docs visibility
    cors_origins: list[str] = ["*"]
    internal_service_token: str | None = None
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_format: str = "text"  # text | json
    rate_limit_quiz_extract: str = "10/minute"
    rate_limit_tutor_chat: str = "30/minute"
    rate_limit_pipeline: str = "5/minute"
    rate_limit_ask_ai: str = "20/minute"

    # ── Download safety ─────────────────────────────────────
    # empty list = allow any non-private HTTPS host
    download_host_allowlist: list[str] = []
    download_max_bytes: int = 100 * 1024 * 1024  # 100 MB

    # ── MongoDB ─────────────────────────────────────────────
    mongo_url: str | None = None

    # ── LLM ─────────────────────────────────────────────────
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    exercise_llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EXERCISE_LLM_MODEL", "ADAPTIVE_EXERCISE_LLM_MODEL"),
    )
    llm_embedding_model: str = "text-embedding-3-small"
    llm_timeout_sec: float = 150
    llm_max_retries: int = 2
    llm_max_workers: int = Field(
        default=8,
        validation_alias=AliasChoices("LLM_MAX_WORKERS", "ADAPTIVE_LLM_MAX_WORKERS"),
    )
    llm_max_concurrency: int | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_MAX_CONCURRENCY", "ADAPTIVE_LLM_MAX_CONCURRENCY"),
    )
    llm_request_timeout_sec: float = Field(
        default=120,
        validation_alias=AliasChoices("LLM_REQUEST_TIMEOUT_SEC", "ADAPTIVE_LLM_TIMEOUT_SEC"),
    )
    llm_prefetch_timeout_sec: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_PREFETCH_TIMEOUT_SEC", "ADAPTIVE_PREFETCH_LLM_TIMEOUT_SEC"
        ),
    )
    llm_retry_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices("LLM_RETRY_ATTEMPTS", "ADAPTIVE_LLM_RETRY_ATTEMPTS"),
    )
    llm_retry_backoff_sec: float = Field(
        default=1.0,
        validation_alias=AliasChoices("LLM_RETRY_BACKOFF_SEC", "ADAPTIVE_LLM_RETRY_BACKOFF_SEC"),
    )

    # ── Content Pipeline ───────────────────────────────────
    embedding_model: str = "keepitreal/vietnamese-sbert"
    embedding_batch_size: int = 32
    use_vi_tokenizer: bool = False
    max_seq_length: int | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    prs_threshold: float = 0.75
    adaptive_mastery_threshold: float = 0.75
    similarity_threshold: float = 0.9
    adaptive_exercise_recent_same_concept_limit: int = Field(
        default=5,
        validation_alias=AliasChoices(
            "ADAPTIVE_EXERCISE_RECENT_SAME_CONCEPT_LIMIT",
            "EXERCISE_RECENT_SAME_CONCEPT_LIMIT",
        ),
    )
    pdf_ocr_concurrency: int = 5
    vision_pdf_request_timeout_sec: float = 120
    vision_agent_api_key: str | None = None
    content_pipeline_job_timeout_sec: float = 1800
    content_pipeline_stage_timeout_sec: float = 300
    content_pipeline_graph_cycle_timeout_sec: float = 900

    # ── S3 Cache ────────────────────────────────────────────
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_bucket_name: str | None = None

    # ── LangChain / LangSmith ──────────────────────────────
    langchain_tracing_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"),
    )
    langchain_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"),
    )
    langchain_project: str = Field(
        default="rinkuzu-ai-api",
        validation_alias=AliasChoices("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT"),
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"),
    )

    # ── Google/Gemini (legacy) ──────────────────────────────
    google_api_key: str | None = None
    gemini_api_key: str | None = None

    @property
    def s3_available(self) -> bool:
        return all([self.s3_endpoint_url, self.s3_access_key_id, self.s3_secret_access_key])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level alias for code importing `api.config.settings` directly.
settings = get_settings()
