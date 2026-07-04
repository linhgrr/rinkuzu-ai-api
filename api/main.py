"""
main.py — FastAPI app entry point for Adaptive Learning Demo.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

from bson import ObjectId
from fastapi import FastAPI
from fastapi.encoders import ENCODERS_BY_TYPE
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import ORJSONResponse
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .config import Settings, get_settings
from .core.content_pipeline.application.pipeline_runner import PipelineRunner
from .core.content_pipeline.application.pipeline_service import PipelineService
from .core.content_pipeline.application.recovery import PipelineJanitor
from .core.content_pipeline.application.source_fetch import download_source_to_dir
from .core.content_pipeline.application.stages.execution import shutdown_pipeline_executor
from .core.content_pipeline.application.stages.model_worker import (
    shutdown_sentence_transformer_worker,
)
from .core.content_pipeline.domain.jobs import PipelineJob
from .core.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient
from .core.content_pipeline.infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_ERROR,
    CONTENT_PROCESSOR_SRC,
)
from .core.content_pipeline.infrastructure.storage.chunk_chroma_store import ChunkChromaStore
from .core.learning.exercise_service import ExerciseService
from .core.learning.session import SessionManager
from .core.shared import mongo_store
from .core.shared.persistence import (
    load_pipeline_job,
    load_pipeline_job_cancel_requested,
    save_pipeline_job,
)
from .core.shared.persistence.pipeline_jobs import list_active_pipeline_jobs
from .exceptions import error_json_response, register_exception_handlers
from .middleware.request_context import RequestContextMiddleware
from .observability import setup_otel, shutdown_otel
from .rate_limit import limiter
from .routers import history as history_router
from .routers import knowledge as knowledge_router
from .routers import pipeline as pipeline_router
from .routers import quiz_drafts as quiz_drafts_router
from .routers import quiz_tutor as quiz_tutor_router
from .routers import session as session_router
from .schemas.common import InfoResponse, ReadinessResponse, StandardResponse, ok

_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
_UPLOAD_MAX_AGE_SECS = 86400

ENCODERS_BY_TYPE.setdefault(ObjectId, str)


def _configure_logging() -> None:
    settings = get_settings()
    logger.remove()
    if settings.log_format == "json":
        logger.add(sys.stdout, serialize=True, level=settings.log_level)
    else:
        logger.add(sys.stdout, level=settings.log_level)


def _configure_llm_tracing() -> None:
    settings = get_settings()
    if not settings.langchain_tracing_v2:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    logger.info("[LangSmith] Tracing ENABLED for project: {}", settings.langchain_project)


def _load_models(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.load_models:
        logger.info("[1/2] Model loading DISABLED — skipping")
        app.state.session_manager = None
        app.state.exercise_service = None
        return

    logger.info("[1/2] Loading SAINT + DQN models...")
    manager = SessionManager(saint_path=settings.saint_path, dqn_path=settings.dqn_path)
    logger.info("  SAINT loaded: {} concepts", manager.n_concepts)
    logger.info("  DQN loaded: ready for action selection")
    app.state.session_manager = manager
    app.state.exercise_service = ExerciseService(session_manager=manager)


def _init_rag_stores(app: FastAPI) -> None:
    app.state.chunk_chroma_store = ChunkChromaStore(embedding_client=EmbeddingClient())
    logger.info("[RAG] ChunkChromaStore initialized")


def _init_pipeline(app: FastAPI) -> None:
    pipeline_service_ref: PipelineService | None = None

    async def persist_pipeline_job_state(job: Any, status: Any, step: Any, progress: Any) -> None:
        if pipeline_service_ref is None:
            raise RuntimeError("PipelineService is not initialized")
        await pipeline_service_ref.persist_job_state(job, status, step, progress)

    pipeline_runner = PipelineRunner(
        load_job=load_pipeline_job,
        load_cancel_flag=load_pipeline_job_cancel_requested,
        save_job=save_pipeline_job,
        persist_job_state=persist_pipeline_job_state,
        chunk_chroma_store=app.state.chunk_chroma_store,
    )

    async def run_content_pipeline(
        job: PipelineJob,
        file_path: str,
        prs_threshold: float | None,
        min_confidence: float,
        *,
        apply_reduction: bool,
        page_batch_size: int,
    ) -> None:
        await pipeline_runner.run(
            job,
            file_path=file_path,
            prs_threshold=prs_threshold,
            min_confidence=min_confidence,
            apply_reduction=apply_reduction,
            page_batch_size=page_batch_size,
        )

    settings = get_settings()
    pipeline_service = PipelineService(
        save_job=save_pipeline_job,
        run_pipeline=run_content_pipeline,
        max_concurrent_jobs=settings.content_pipeline_max_concurrent_jobs,
    )
    # Rebind so the closure in persist_pipeline_job_state sees the real service.
    pipeline_service_ref = pipeline_service

    app.state.content_pipeline_runner = pipeline_runner
    app.state.content_pipeline_service = pipeline_service
    app.state.pipeline_janitor = _build_pipeline_janitor(pipeline_service, settings)


def _build_pipeline_janitor(
    pipeline_service: PipelineService, settings: Settings
) -> PipelineJanitor:
    async def _recover() -> None:
        await pipeline_service.recover_interrupted_jobs(
            list_active=list_active_pipeline_jobs,
            download_source=download_source_to_dir,
            recovery_max_age_sec=settings.content_pipeline_recovery_max_age_sec,
        )

    async def _reap() -> None:
        await pipeline_service.reap_stalled_jobs(
            list_active=list_active_pipeline_jobs,
            stalled_after_sec=settings.content_pipeline_job_stalled_after_sec,
        )

    return PipelineJanitor(
        recover=_recover,
        reap=_reap,
        reaper_interval_sec=settings.content_pipeline_reaper_interval_sec,
    )


def _log_llm_config(settings: Settings) -> None:
    from .core.shared.llm import normalize_llm_base_url

    base_url = normalize_llm_base_url(settings.llm_base_url)
    model = settings.llm_model or "(not set)"
    custom_provider = settings.llm_custom_provider or "(auto)"
    api_key = settings.llm_api_key or ""
    _key_prefix_len = 6
    key_display = (
        f"{api_key[:_key_prefix_len]}…"
        if len(api_key) > _key_prefix_len
        else (api_key or "(not set)")
    )
    exercise_model = settings.active_exercise_llm_model or model
    logger.info("[LLM] base_url    = {}", base_url)
    logger.info("[LLM] model       = {}", model)
    logger.info("[LLM] provider    = {}", custom_provider)
    logger.info("[LLM] api_key     = {}", key_display)
    logger.info("[LLM] exercise    = {}", exercise_model)
    logger.info("[LLM] max_retries = {}", settings.llm_max_retries)


def _purge_stale_uploads() -> None:
    purged = 0
    for f in _UPLOAD_DIR.glob("*"):
        if f.is_file() and (time.time() - f.stat().st_mtime) > _UPLOAD_MAX_AGE_SECS:
            try:
                f.unlink()
                purged += 1
            except OSError as exc:
                logger.warning("[Janitor] Failed to purge stale upload {}: {}", f.name, exc)
    if purged:
        logger.info("[Janitor] Purged {} stale upload(s) older than 24 h", purged)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Startup: initialize all components, yield, then shut down."""
    _configure_logging()

    logger.info("=" * 60)
    logger.info("  ALSS-LEPC Full Demo — Starting up...")
    logger.info("=" * 60)

    _configure_llm_tracing()

    settings = get_settings()

    _log_llm_config(settings)
    setup_otel(app, settings)

    logger.info("[0/2] Connecting to MongoDB...")
    await mongo_store.init_mongo(mongodb_uri=settings.mongodb_uri)

    _load_models(app)

    app.state.content_processor_available = CONTENT_PROCESSOR_AVAILABLE
    app.state.content_processor_error = CONTENT_PROCESSOR_ERROR
    app.state.content_processor_src = CONTENT_PROCESSOR_SRC

    _init_rag_stores(app)

    if CONTENT_PROCESSOR_AVAILABLE:
        logger.info("[PipelineRuntime] Content pipeline runtime available")
    else:
        logger.warning(
            "[PipelineRuntime] Content pipeline runtime unavailable: {}",
            CONTENT_PROCESSOR_ERROR or "unknown error",
        )

    _init_pipeline(app)
    _purge_stale_uploads()

    janitor = getattr(app.state, "pipeline_janitor", None)
    if janitor is not None and CONTENT_PROCESSOR_AVAILABLE and mongo_store.is_available():
        await janitor.start()

    logger.info("[2/2] Server ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")
    janitor = getattr(app.state, "pipeline_janitor", None)
    if janitor is not None:
        await janitor.stop()
    pipeline_service = getattr(app.state, "content_pipeline_service", None)
    if pipeline_service is not None:
        await pipeline_service.shutdown(timeout_sec=2.0)
    await shutdown_sentence_transformer_worker()
    shutdown_pipeline_executor(wait=False, cancel_futures=True)
    exercise_service = getattr(app.state, "exercise_service", None)
    if exercise_service:
        exercise_service.close()
    shutdown_otel(app)
    await mongo_store.shutdown_mongo()


settings = get_settings()
_is_dev = settings.environment == "dev"
app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + D3QN RL",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    contact={
        "name": "Rinkuzu Team",
        "url": "https://github.com/rinkuzu/rinkuzu-ai-api",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if _is_dev else None,
    redoc_url="/redoc" if _is_dev else None,
    openapi_url="/openapi.json" if _is_dev else None,
)


def _build_openapi_security() -> tuple[dict[str, dict[str, str]], list[dict[str, list[str]]]]:
    schemes: dict[str, dict[str, str]] = {
        "XUserIdHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-user-id",
            "description": "Authenticated user id forwarded by the frontend proxy.",
        },
    }
    requirement: dict[str, list[str]] = {"XUserIdHeader": []}

    if settings.internal_service_token:
        schemes["XServiceTokenHeader"] = {
            "type": "apiKey",
            "in": "header",
            "name": "x-service-token",
            "description": "Shared internal token used by the frontend proxy when calling the backend API.",
        }
        requirement["XServiceTokenHeader"] = []

    return schemes, [requirement]


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    security_schemes, security = _build_openapi_security()
    components = openapi_schema.setdefault("components", {})
    components.setdefault("securitySchemes", {}).update(security_schemes)
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Same-origin deployment behind the current API host.",
        }
    ]
    openapi_schema["security"] = security

    public_operations = {
        "/api/ready": {"get"},
        "/api/health": {"get"},
        "/api/info": {"get"},
        "/api/pipeline/status": {"get"},
    }
    for path, methods in public_operations.items():
        path_item = openapi_schema.get("paths", {}).get(path, {})
        for method in methods:
            operation = path_item.get(method)
            if operation is not None:
                operation["security"] = []

    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

register_exception_handlers(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast("Any", _rate_limit_exceeded_handler))

# Middleware — outermost first (last added = outermost)
app.add_middleware(RequestContextMiddleware)

_cors_origins = list(settings.cors_origins)
_cors_allow_credentials = True
if "*" in _cors_origins:
    if settings.environment == "prod":
        raise RuntimeError(
            "CORS_ORIGINS cannot be ['*'] in production — set an explicit allowlist."
        )
    # Browsers reject `Access-Control-Allow-Origin: *` paired with credentials. Drop
    # credentials so the dev wildcard does not produce an invalid header combination.
    _cors_allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router.router)
app.include_router(knowledge_router.router)
app.include_router(pipeline_router.router)
app.include_router(history_router.router)
app.include_router(quiz_drafts_router.router)
app.include_router(quiz_tutor_router.router)


@app.get("/api/live", include_in_schema=False)
async def liveness() -> Any:
    """Kubernetes liveness probe — always 200 while the process is running."""
    return ok({"status": "ok"})


def _build_readiness_payload() -> tuple[dict, bool]:
    cfg = get_settings()
    models_loaded = getattr(app.state, "session_manager", None) is not None
    models_ready = models_loaded if cfg.load_models else True
    mongo_available = mongo_store.is_available()
    pipeline_service_ready = getattr(app.state, "content_pipeline_service", None) is not None
    content_pipeline_available = bool(getattr(app.state, "content_processor_available", False))
    ready = mongo_available and models_ready and pipeline_service_ready
    payload = {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "mongo_available": mongo_available,
        "models_enabled": cfg.load_models,
        "models_loaded": models_loaded,
        "content_pipeline_available": content_pipeline_available,
        "content_pipeline_service_ready": pipeline_service_ready,
    }
    return payload, ready


@app.get("/api/ready", response_model=StandardResponse[ReadinessResponse])
async def readiness() -> Any:
    """Kubernetes readiness probe — 503 until all dependencies are up."""
    payload, ready = _build_readiness_payload()
    if not ready:
        return error_json_response(
            code="service_unavailable",
            message="Service unavailable",
            detail="Adaptive API is not ready",
            status_code=503,
            meta=payload,
        )
    return ok(payload)


@app.get("/api/health", response_model=StandardResponse[ReadinessResponse])
async def health() -> Any:
    """Backwards-compat alias for /api/ready."""
    payload, ready = _build_readiness_payload()
    if not ready:
        return error_json_response(
            code="service_unavailable",
            message="Service unavailable",
            detail="Adaptive API is not ready",
            status_code=503,
            meta=payload,
        )
    return ok(payload)


@app.get("/api/info", response_model=StandardResponse[InfoResponse])
async def info() -> Any:
    cfg = get_settings()
    manager = getattr(app.state, "session_manager", None)
    return ok(
        {
            "models_enabled": cfg.load_models,
            "models_loaded": manager is not None,
            "n_concepts": manager.n_concepts if manager else 0,
            "mongo_available": mongo_store.is_available(),
            "content_pipeline_available": bool(
                getattr(app.state, "content_processor_available", False)
            ),
        }
    )
