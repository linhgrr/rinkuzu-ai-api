"""
main.py — FastAPI app entry point for Adaptive Learning Demo.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .config import Settings, get_settings
from .core.content_pipeline.application.pipeline_runner import PipelineRunner
from .core.content_pipeline.application.pipeline_service import PipelineService
from .core.content_pipeline.application.stages.execution import shutdown_pipeline_executor
from .core.content_pipeline.application.stages.model_worker import shutdown_sentence_transformer_worker
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
from .exceptions import register_exception_handlers
from .middleware.request_context import RequestContextMiddleware
from .rate_limit import limiter
from .routers import history as history_router
from .routers import knowledge as knowledge_router
from .routers import pipeline as pipeline_router
from .routers import quiz_drafts as quiz_drafts_router
from .routers import quiz_tutor as quiz_tutor_router
from .routers import session as session_router

_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
_UPLOAD_MAX_AGE_SECS = 86400


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
    db = mongo_store._get_db() if mongo_store.is_available() else None
    app.state.document_chunks_col = db["al_document_chunks"] if db is not None else None
    logger.info("[RAG] ChunkChromaStore and MongoDB collection initialized")


def _init_pipeline(app: FastAPI) -> None:
    pipeline_service_ref: PipelineService | None = None

    async def persist_pipeline_job_state(job, status, step, progress) -> None:
        if pipeline_service_ref is None:
            raise RuntimeError("PipelineService is not initialized")
        await pipeline_service_ref.persist_job_state(job, status, step, progress)

    pipeline_runner = PipelineRunner(
        load_job=mongo_store.require_pipeline_repo().load,
        save_job=mongo_store.require_pipeline_repo().save,
        persist_job_state=persist_pipeline_job_state,
        chunk_chroma_store=app.state.chunk_chroma_store,
        document_chunks_col=app.state.document_chunks_col,
    )

    async def run_content_pipeline(
        job,
        file_path,
        prs_threshold,
        min_confidence,
        *,
        apply_reduction,
        page_batch_size,
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
        save_job=mongo_store.require_pipeline_repo().save,
        run_pipeline=run_content_pipeline,
        max_concurrent_jobs=settings.content_pipeline_max_concurrent_jobs,
    )
    # Rebind so the closure in persist_pipeline_job_state sees the real service.
    pipeline_service_ref = pipeline_service

    app.state.content_pipeline_runner = pipeline_runner
    app.state.content_pipeline_service = pipeline_service


def _log_llm_config(settings: Settings) -> None:
    from .core.content_pipeline.infrastructure.llm.openai_responses import (  # noqa: PLC0415
        normalize_openai_base_url,
    )

    base_url = normalize_openai_base_url(settings.openai_base_url)
    model = settings.openai_model or "(not set)"
    api_key = settings.openai_api_key or ""
    _key_prefix_len = 6
    key_display = f"{api_key[:_key_prefix_len]}…" if len(api_key) > _key_prefix_len else (api_key or "(not set)")
    exercise_model = settings.exercise_llm_model or model
    logger.info("[LLM] base_url    = {}", base_url)
    logger.info("[LLM] model       = {}", model)
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
async def lifespan(app: FastAPI):
    """Startup: initialize all components, yield, then shut down."""
    _configure_logging()

    logger.info("=" * 60)
    logger.info("  ALSS-LEPC Full Demo — Starting up...")
    logger.info("=" * 60)

    _configure_llm_tracing()

    settings = get_settings()

    _log_llm_config(settings)

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

    logger.info("[2/2] Server ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")
    pipeline_service = getattr(app.state, "content_pipeline_service", None)
    if pipeline_service is not None:
        await pipeline_service.shutdown(timeout_sec=2.0)
    await shutdown_sentence_transformer_worker()
    shutdown_pipeline_executor(wait=False, cancel_futures=True)
    exercise_service = getattr(app.state, "exercise_service", None)
    if exercise_service:
        exercise_service.close()


settings = get_settings()
_is_dev = settings.environment == "dev"
app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + D3QN RL",
    version="1.0.0",
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
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
async def liveness():
    """Kubernetes liveness probe — always 200 while the process is running."""
    return {"status": "ok"}


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


@app.get("/api/ready")
async def readiness():
    """Kubernetes readiness probe — 503 until all dependencies are up."""
    payload, ready = _build_readiness_payload()
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/api/health")
async def health():
    """Backwards-compat alias for /api/ready."""
    payload, ready = _build_readiness_payload()
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/api/info")
async def info():
    cfg = get_settings()
    manager = getattr(app.state, "session_manager", None)
    return {
        "models_enabled": cfg.load_models,
        "models_loaded": manager is not None,
        "n_concepts": manager.n_concepts if manager else 0,
        "mongo_available": mongo_store.is_available(),
        "content_pipeline_available": bool(
            getattr(app.state, "content_processor_available", False)
        ),
    }
