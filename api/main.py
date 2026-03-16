"""
main.py — FastAPI app entry point for Adaptive Learning Demo.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import get_settings
from .exceptions import register_exception_handlers
from .core.session import SessionManager
from .core.exercise_gen import init_llm
from .core import mongo_store
from .services.exercise_service import ExerciseService
from .routers import session as session_router
from .routers import knowledge as knowledge_router
from .routers import pipeline as pipeline_router
from .routers import history as history_router
from .routers import quiz_extract as quiz_extract_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, init components."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("  ALSS-LEPC Full Demo — Starting up...")
    logger.info("=" * 60)

    # Init LLM
    init_llm(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
    )

    # Init MongoDB
    logger.info("[0/2] Connecting to MongoDB...")
    await mongo_store.init_mongo(mongo_url=settings.mongo_url)

    # Create session repo reference for ExerciseService
    session_repo = mongo_store.get_session_repo()

    if settings.load_models:
        logger.info("[1/2] Loading SAINT + DQN models...")
        manager = SessionManager(
            saint_path=settings.saint_path,
            dqn_path=settings.dqn_path,
        )
        logger.info(f"  SAINT loaded: {manager.n_concepts} concepts")
        logger.info("  DQN loaded: ready for action selection")

        # Create ExerciseService with repository dependency
        exercise_service = ExerciseService(session_repo=session_repo)
    else:
        logger.info("[1/2] Model loading DISABLED — skipping")
        manager = None
        exercise_service = None

    # Store in app state — accessed by dependencies.py
    app.state.session_manager = manager
    app.state.exercise_service = exercise_service

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/api/info")
async def info():
    manager = app.state.session_manager
    if not manager:
        return {"error": "Models not loaded"}
    return {
        "n_concepts": manager.n_concepts,
        "concept_names": {
            cid: name for cid, name in list(manager.concept_names.items())[:20]
        },
        "models": {
            "saint": "saint_best.pt",
            "dqn": "dqn_best.pt",
        },
    }
