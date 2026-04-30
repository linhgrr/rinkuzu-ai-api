"""
main.py — FastAPI app entry point for Adaptive Learning Demo.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import get_settings
from .core.content_pipeline.application.pipeline_runner import PipelineRunner
from .core.content_pipeline.application.pipeline_service import PipelineService
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
from .core.shared.llm import initialize_shared_llm
from .exceptions import register_exception_handlers
from .routers import history as history_router
from .routers import knowledge as knowledge_router
from .routers import pipeline as pipeline_router
from .routers import quiz_extract as quiz_extract_router
from .routers import quiz_tutor as quiz_tutor_router
from .routers import session as session_router

_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
_UPLOAD_MAX_AGE_SECS = 86400


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
        load_job=mongo_store.load_pipeline_job,
        save_job=mongo_store.save_pipeline_job,
        persist_job_state=persist_pipeline_job_state,
        chunk_chroma_store=app.state.chunk_chroma_store,
        document_chunks_col=app.state.document_chunks_col,
    )

    async def run_content_pipeline(job, file_path, prs_threshold, min_confidence, apply_reduction) -> None:
        await pipeline_runner.run(
            job,
            file_path=file_path,
            prs_threshold=prs_threshold,
            min_confidence=min_confidence,
            apply_reduction=apply_reduction,
        )

    pipeline_service = PipelineService(
        save_job=mongo_store.save_pipeline_job,
        run_pipeline=run_content_pipeline,
    )
    # Rebind so the closure in persist_pipeline_job_state sees the real service.
    pipeline_service_ref = pipeline_service  # type: ignore[assignment]

    app.state.content_pipeline_runner = pipeline_runner
    app.state.content_pipeline_service = pipeline_service


def _purge_stale_uploads() -> None:
    purged = 0
    for f in _UPLOAD_DIR.glob("*"):
        if f.is_file() and (time.time() - f.stat().st_mtime) > _UPLOAD_MAX_AGE_SECS:
            try:
                f.unlink()
                purged += 1
            except OSError:
                pass
    if purged:
        logger.info("[Janitor] Purged {} stale upload(s) older than 24 h", purged)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize all components, yield, then shut down."""
    logger.info("=" * 60)
    logger.info("  ALSS-LEPC Full Demo — Starting up...")
    logger.info("=" * 60)

    _configure_llm_tracing()

    settings = get_settings()
    initialize_shared_llm(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
    )

    logger.info("[0/2] Connecting to MongoDB...")
    await mongo_store.init_mongo(mongo_url=settings.mongo_url)

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
    exercise_service = getattr(app.state, "exercise_service", None)
    if exercise_service:
        exercise_service.close()


settings = get_settings()
_is_dev = settings.environment == "dev"
app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + D3QN RL",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _is_dev else None,
    redoc_url="/redoc" if _is_dev else None,
    openapi_url="/openapi.json" if _is_dev else None,
)

register_exception_handlers(app)

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
app.include_router(quiz_extract_router.router)
app.include_router(quiz_tutor_router.router)


@app.get("/api/health")
async def health():
    cfg = get_settings()
    models_loaded = getattr(app.state, "session_manager", None) is not None
    models_ready = models_loaded if cfg.load_models else True
    mongo_available = mongo_store.is_available()
    pipeline_service_ready = getattr(app.state, "content_pipeline_service", None) is not None
    content_pipeline_available = bool(getattr(app.state, "content_processor_available", False))
    ready = mongo_available and models_ready and pipeline_service_ready

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "mongo_available": mongo_available,
        "models_enabled": cfg.load_models,
        "models_loaded": models_loaded,
        "content_pipeline_available": content_pipeline_available,
        "content_pipeline_service_ready": pipeline_service_ready,
    }


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
