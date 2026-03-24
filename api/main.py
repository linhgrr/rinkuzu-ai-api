"""
main.py — FastAPI app entry point for Adaptive Learning Demo.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import get_settings
from .exceptions import register_exception_handlers
from .core.learning.session import SessionManager
from .core.shared.llm import initialize_shared_llm
from .core.shared import mongo_store
from .core.content_pipeline.application.pipeline_runner import PipelineRunner
from .core.content_pipeline.application.pipeline_service import PipelineService
from .core.content_pipeline.infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_ERROR,
    CONTENT_PROCESSOR_SRC,
)
from .core.learning.exercise_service import ExerciseService
from .routers import session as session_router
from .routers import knowledge as knowledge_router
from .routers import pipeline as pipeline_router
from .routers import history as history_router
from .routers import quiz_extract as quiz_extract_router
from .routers import quiz_tutor as quiz_tutor_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, init components."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("  ALSS-LEPC Full Demo — Starting up...")
    logger.info("=" * 60)

    # Init LLM Tracing & Connectivity
    import os
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key or ""
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        logger.info(f"[LangSmith] Tracing ENABLED for project: {settings.langchain_project}")

    initialize_shared_llm(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
    )

    # Init MongoDB
    logger.info("[0/2] Connecting to MongoDB...")
    await mongo_store.init_mongo(mongo_url=settings.mongo_url)

    if settings.load_models:
        logger.info("[1/2] Loading SAINT + DQN models...")
        manager = SessionManager(
            saint_path=settings.saint_path,
            dqn_path=settings.dqn_path,
        )
        logger.info(f"  SAINT loaded: {manager.n_concepts} concepts")
        logger.info("  DQN loaded: ready for action selection")

        # Create ExerciseService with repository dependency
        exercise_service = ExerciseService(session_manager=manager)
    else:
        logger.info("[1/2] Model loading DISABLED — skipping")
        manager = None
        exercise_service = None

    # Store in app state — accessed by dependencies.py
    app.state.session_manager = manager
    app.state.exercise_service = exercise_service
    app.state.content_processor_available = CONTENT_PROCESSOR_AVAILABLE
    app.state.content_processor_error = CONTENT_PROCESSOR_ERROR
    app.state.content_processor_src = CONTENT_PROCESSOR_SRC

    if CONTENT_PROCESSOR_AVAILABLE:
        logger.info("[PipelineRuntime] Content pipeline runtime available")
    else:
        logger.warning(
            "[PipelineRuntime] Content pipeline runtime unavailable: {}",
            CONTENT_PROCESSOR_ERROR or "unknown error",
        )

    pipeline_service: PipelineService | None = None

    async def persist_pipeline_job_state(job, status, step, progress):
        if pipeline_service is None:
            raise RuntimeError("PipelineService is not initialized")
        await pipeline_service.persist_job_state(job, status, step, progress)

    pipeline_runner = PipelineRunner(
        load_job=mongo_store.load_pipeline_job,
        save_job=mongo_store.save_pipeline_job,
        persist_job_state=persist_pipeline_job_state,
    )

    async def run_content_pipeline(job, file_path, prs_threshold, min_confidence, apply_reduction):
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
    app.state.content_pipeline_runner = pipeline_runner
    app.state.content_pipeline_service = pipeline_service

    logger.info("[2/2] Server ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")
    if exercise_service:
        exercise_service.close()


app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + D3QN RL",
    version="1.0.0",
    lifespan=lifespan,
)

# Register custom exception handlers
register_exception_handlers(app)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(session_router.router)
app.include_router(knowledge_router.router)
app.include_router(pipeline_router.router)
app.include_router(history_router.router)
app.include_router(quiz_extract_router.router)
app.include_router(quiz_tutor_router.router)


@app.get("/api/health")
async def health():
    settings = get_settings()
    models_loaded = getattr(app.state, "session_manager", None) is not None
    models_ready = models_loaded if settings.load_models else True
    mongo_available = mongo_store.is_available()
    pipeline_service_ready = getattr(app.state, "content_pipeline_service", None) is not None
    content_pipeline_available = bool(getattr(app.state, "content_processor_available", False))
    ready = mongo_available and models_ready and pipeline_service_ready

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "mongo_available": mongo_available,
        "models_enabled": settings.load_models,
        "models_loaded": models_loaded,
        "content_pipeline_available": content_pipeline_available,
        "content_pipeline_service_ready": pipeline_service_ready,
    }


@app.get("/api/info")
async def info():
    settings = get_settings()
    manager = app.state.session_manager
    models_loaded = manager is not None

    return {
        "models_enabled": settings.load_models,
        "models_loaded": models_loaded,
        "n_concepts": manager.n_concepts if manager else 0,
        "mongo_available": mongo_store.is_available(),
        "content_pipeline_available": bool(
            getattr(app.state, "content_processor_available", False)
        ),
    }
