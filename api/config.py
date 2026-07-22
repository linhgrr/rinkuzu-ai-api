"""
config.py — Centralized application settings using Pydantic BaseSettings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Placeholder markers (case-insensitive). Never echo caller input in validation errors.
_SERVICE_TOKEN_PLACEHOLDER_MARKERS = frozenset(
    {
        "replace-me",
        "change-me",
        "your-token",
        "example",
        "default",
    }
)
_MIN_SERVICE_TOKEN_LENGTH = 32
_MAX_REPEATED_CYCLE_LEN = 8
_ENVIRONMENT_ALIASES = {
    "dev": "dev",
    "development": "dev",
    "staging": "staging",
    "prod": "prod",
    "production": "prod",
}


def _is_placeholder_or_repeated_token(token: str) -> bool:
    """True for obvious placeholder / default / repeated tokens (case-insensitive).

    Detected before the minimum-length check so short markers still surface as
    placeholders. Long (>=32) values built from the same markers, all-same
    characters, or short pure cycles are also rejected. Does not echo ``token``.
    """
    folded = token.casefold()
    if not folded:
        return False

    if folded in _SERVICE_TOKEN_PLACEHOLDER_MARKERS:
        return True

    # All characters identical (e.g. "aaaaaaaa...").
    if len(set(folded)) == 1:
        return True

    # Pure short-cycle repetition (e.g. "ababab...", "xyzxyzxyz...").
    n = len(folded)
    max_period = min(_MAX_REPEATED_CYCLE_LEN, n // 2)
    for period in range(1, max_period + 1):
        if n % period == 0 and folded == folded[:period] * (n // period):
            return True

    # Marker repeated / packed to any length (including >=32).
    for marker in _SERVICE_TOKEN_PLACEHOLDER_MARKERS:
        if marker not in folded:
            continue
        # Exact k-fold concatenation of the marker.
        if (
            len(folded) >= len(marker)
            and len(folded) % len(marker) == 0
            and folded == marker * (len(folded) // len(marker))
        ):
            return True
        # Truncated packing: marker * ceil(n/len(marker)) clipped to n.
        reps = (len(folded) // len(marker)) + 1
        if folded == (marker * reps)[:n]:
            return True
        # Marker-only content with optional separators after stripping markers.
        remainder = folded.replace(marker, "")
        if remainder == "" or set(remainder) <= {"-", "_", ".", " "}:
            return True

    return False


def normalize_endpoint(value: str | None, *, default_scheme: str) -> str | None:
    """Normalize a host/URL into a scheme-qualified endpoint (or None if empty)."""
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
        # Never echo secrets (service tokens, API keys) in ValidationError payloads.
        hide_input_in_errors=True,
    )

    # ── Models ──────────────────────────────────────────────
    load_models: bool = True
    saint_path: str = str(BASE_DIR / "models" / "saint_best.pt")
    dqn_path: str = str(BASE_DIR / "models" / "dqn_best.pt")
    mlp_weights_path: str = str(BASE_DIR / "models" / "prereq_vimath_bgem3_namedef_concat_rich.pth")

    # ChromaDB persistence dir. Kept under api/core/ (its historical home) so the
    # existing on-disk store survives the domains/ refactor; override via env.
    chroma_persist_dir: str = str(BASE_DIR / "api" / "core" / "chroma_db")

    # ── App Config ──────────────────────────────────────────
    environment: str = "dev"  # dev | staging | prod — controls docs visibility
    cors_origins: list[str] = ["*"]
    # Required in prod; optional in dev/staging so local tests can construct Settings.
    # Request-time auth still fails closed when the token is unset.
    internal_service_token: str | None = None
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_format: str = "text"  # text | json
    otel_enabled: bool = False
    otel_service_name: str = "rinkuzu-ai-api"
    rate_limit_pipeline: str = "5/minute"
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

    # ── LLM pricing (USD per 1M tokens) ─────────────────────
    # Defaults from DeepSeek V4 pricing (cache-miss input). Model name is
    # matched by substring: "pro" → Pro tier, else Flash.
    llm_price_flash_input_per_m: float = 0.14
    llm_price_flash_output_per_m: float = 0.28
    llm_price_pro_input_per_m: float = 0.435
    llm_price_pro_output_per_m: float = 0.87
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
    prs_threshold: float | None = None  # None = use threshold from ViMath checkpoint metadata
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
    content_pipeline_max_pdf_pages: int = Field(
        default=100,
        validation_alias="CONTENT_PIPELINE_MAX_PDF_PAGES",
    )
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
    content_pipeline_debug_artifact_max_chars: int = 80_000
    # ── Pipeline resilience (reaper / recovery / dedup) ───────
    content_pipeline_reaper_interval_sec: int = 60
    content_pipeline_job_stalled_after_sec: int = 900
    content_pipeline_recovery_max_age_sec: int = 3600
    content_pipeline_dedup_window_sec: int = 30
    content_pipeline_max_retry_count: int = 3

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
    ocr_key_encryption_secret: str | None = Field(
        default=None,
        validation_alias="OCR_KEY_ENCRYPTION_SECRET",
    )
    ocr_timeout_sec: float = Field(default=120, validation_alias="OCR_TIMEOUT_SEC")

    # ── Quiz extraction ────────────────────────────────────
    quiz_extract_max_pdf_bytes: int = Field(
        default=5 * 1024 * 1024,
        validation_alias="QUIZ_EXTRACT_MAX_PDF_BYTES",
    )
    quiz_extract_max_chars: int = Field(
        default=200_000,
        validation_alias="QUIZ_EXTRACT_MAX_CHARS",
    )
    quiz_extract_source_download_timeout_sec: float = Field(
        default=180,
        validation_alias="QUIZ_EXTRACT_SOURCE_DOWNLOAD_TIMEOUT_SEC",
    )
    quiz_extract_source_endpoint_timeout_sec: float = Field(
        default=75,
        validation_alias="QUIZ_EXTRACT_SOURCE_ENDPOINT_TIMEOUT_SEC",
    )

    # ── S3 Cache ────────────────────────────────────────────
    object_storage_region: str = "ap-southeast-1"
    object_storage_endpoint_internal: str | None = None
    object_storage_endpoint_external: str | None = None
    object_storage_access_key: str | None = None
    object_storage_secret_key: str | None = None
    object_storage_bucket: str | None = None
    object_storage_addressing_style: str = "path"
    object_storage_quiz_connect_timeout_sec: float = 10
    object_storage_quiz_read_timeout_sec: float = 60
    object_storage_quiz_retry_attempts: int = 2

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

    @field_validator("internal_service_token", mode="before")
    @classmethod
    def normalize_internal_service_token(cls, value: object) -> str | None:
        """Trim and validate service tokens without echoing the input value.

        Order: trim → empty→None → placeholder/repeated → min-length. Non-prod
        may omit the token (None); prod is enforced by the model validator.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            # Pydantic v2 no longer wraps TypeError raised by validators.
            raise ValueError("internal service token must be a string")  # noqa: TRY004
        token = value.strip()
        if not token:
            return None
        # Placeholder / repeated checks before min-length so short markers and
        # long packed markers share one clear error path (never echo input).
        if _is_placeholder_or_repeated_token(token):
            raise ValueError("internal service token must not be a placeholder value")
        if len(token) < _MIN_SERVICE_TOKEN_LENGTH:
            raise ValueError(
                f"internal service token must be at least {_MIN_SERVICE_TOKEN_LENGTH} characters"
            )
        return token

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> str:
        """Normalize supported environment aliases to one canonical value."""
        if not isinstance(value, str):
            # Pydantic v2 no longer wraps TypeError raised by validators.
            raise ValueError("environment must be a string")  # noqa: TRY004
        normalized = value.strip().casefold()
        try:
            return _ENVIRONMENT_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError("environment must be dev, staging, or prod") from exc

    @model_validator(mode="after")
    def require_internal_service_token_in_prod(self) -> Self:
        """Prod must fail closed at startup when the service token is unusable."""
        if self.environment == "prod" and not self.internal_service_token:
            raise ValueError("internal service token is required in production")
        return self

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
        internal = normalize_endpoint(
            self.object_storage_endpoint_internal,
            default_scheme="http",
        )
        external = self.object_storage_public_base_url
        if self.environment == "dev":
            return external or internal
        return internal or external

    @property
    def object_storage_public_base_url(self) -> str | None:
        return normalize_endpoint(
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
