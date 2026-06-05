"""
config.py — Centralized application settings using Pydantic BaseSettings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


def _normalize_endpoint(value: str | None, *, default_scheme: str) -> str | None:
    raw = (value or "").strip().rstrip("/")
    if not raw:
        return None
    if "://" in raw:
        return raw
    return f"{default_scheme}://{raw}"


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
    mlp_weights_path: str = str(BASE_DIR / "models" / "prereq_mlp.pth")

    # ── App Config ──────────────────────────────────────────
    environment: str = "dev"  # dev | staging | prod — controls docs visibility
    cors_origins: list[str] = ["*"]
    internal_service_token: str | None = None
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_format: str = "text"  # text | json
    otel_enabled: bool = False
    otel_service_name: str = "rinkuzu-ai-api"
    rate_limit_tutor_chat: str = "30/minute"
    rate_limit_pipeline: str = "5/minute"
    rate_limit_ask_ai: str = "20/minute"
    rate_limit_session: str = "30/minute"
    rate_limit_quiz_drafts: str = "20/minute"
    rate_limit_history: str = "30/minute"

    # ── Download safety ─────────────────────────────────────
    # empty list = allow any non-private HTTPS host
    download_host_allowlist: list[str] = []
    download_max_bytes: int = 100 * 1024 * 1024  # 100 MB

    # ── MongoDB ─────────────────────────────────────────────
    mongodb_uri: str | None = None

    # ── LLM ─────────────────────────────────────────────────
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    llm_model: str | None = Field(default=None, validation_alias="LLM_MODEL")
    llm_custom_provider: str | None = Field(
        default=None,
        validation_alias="LLM_CUSTOM_PROVIDER",
    )
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
    content_pipeline_max_concurrent_jobs: int = 2
    embedding_model: str = "keepitreal/vietnamese-sbert"
    embedding_batch_size: int = 32
    use_vi_tokenizer: bool = False
    max_seq_length: int | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    prs_threshold: float = (
        0.5  # MLP probability threshold (0.5 matches LectureBank F1=0.825 evaluation)
    )
    adaptive_mastery_threshold: float = 0.75
    similarity_threshold: float = 0.9
    adaptive_exercise_recent_same_concept_limit: int = Field(
        default=5,
        validation_alias=AliasChoices(
            "ADAPTIVE_EXERCISE_RECENT_SAME_CONCEPT_LIMIT",
            "EXERCISE_RECENT_SAME_CONCEPT_LIMIT",
        ),
    )
    content_pipeline_job_timeout_sec: float = 1800
    content_pipeline_stage_timeout_sec: float = 300
    content_pipeline_graph_cycle_timeout_sec: float = 900
    content_pipeline_pdf_page_batch_size: int = 10
    content_pipeline_pdf_batch_max_bytes: int = 4 * 1024 * 1024
    content_pipeline_batch_failure_ratio_threshold: float = 0.5
    content_pipeline_llm_request_timeout_sec: float = Field(
        default=180,
        validation_alias=AliasChoices(
            "CONTENT_PIPELINE_LLM_REQUEST_TIMEOUT_SEC",
            "CONTENT_PIPELINE_RESPONSES_TIMEOUT_SEC",
        ),
    )
    content_pipeline_llm_retry_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "CONTENT_PIPELINE_LLM_RETRY_ATTEMPTS",
            "LLM_RETRY_ATTEMPTS",
            "ADAPTIVE_LLM_RETRY_ATTEMPTS",
        ),
    )
    content_pipeline_llm_retry_backoff_sec: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "CONTENT_PIPELINE_LLM_RETRY_BACKOFF_SEC",
            "LLM_RETRY_BACKOFF_SEC",
            "ADAPTIVE_LLM_RETRY_BACKOFF_SEC",
        ),
    )
    content_pipeline_cache_restore_timeout_sec: float = 10.0
    content_pipeline_extraction_secs_per_page: float = 20.0
    content_pipeline_default_retry_after_sec: int = 3
    content_pipeline_active_retry_after_sec: int = 5
    content_pipeline_long_stage_retry_after_sec: int = 10
    content_pipeline_delayed_retry_after_sec: int = 15
    content_pipeline_job_delayed_after_sec: int = 360

    # ── OCR API ────────────────────────────────────────────
    ocr_base_url: str = Field(
        default="https://api.va.landing.ai/v1/ade/parse",
        validation_alias="OCR_BASE_URL",
    )
    ocr_model: str = Field(
        default="dpt-2-mini",
        validation_alias="OCR_MODEL",
    )
    ocr_api_key: str | None = Field(default=None, validation_alias="OCR_API_KEY")
    ocr_timeout_sec: float = Field(default=120, validation_alias="OCR_TIMEOUT_SEC")

    # ── S3 Cache ────────────────────────────────────────────
    object_storage_region: str = "ap-southeast-1"
    object_storage_endpoint_internal: str | None = None
    object_storage_endpoint_external: str | None = None
    object_storage_access_key: str | None = None
    object_storage_secret_key: str | None = None
    object_storage_bucket: str | None = None
    object_storage_addressing_style: str = "path"

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
        return all(
            [
                self.object_storage_client_endpoint,
                self.object_storage_access_key,
                self.object_storage_secret_key,
                self.object_storage_bucket,
            ]
        )

    @property
    def object_storage_client_endpoint(self) -> str | None:
        internal = _normalize_endpoint(
            self.object_storage_endpoint_internal,
            default_scheme="http",
        )
        external = self.object_storage_public_base_url
        if self.environment == "dev":
            return external or internal
        return internal or external

    @property
    def object_storage_public_base_url(self) -> str | None:
        return _normalize_endpoint(
            self.object_storage_endpoint_external,
            default_scheme="https",
        )

    @property
    def active_exercise_llm_model(self) -> str | None:
        return self.exercise_llm_model or self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level alias for code importing `api.config.settings` directly.
settings = get_settings()
