"""
main.py — FastAPI app entry point for Adaptive Learning Demo
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

from .core.session import SessionManager
from .core.exercise_gen import init_llm
from .core import mongo_store
from .routers import session as session_router
from .routers import knowledge as knowledge_router
from .routers import pipeline as pipeline_router
from .routers import history as history_router

# Paths to pre-trained models
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
SAINT_PATH = str(MODELS_DIR / "saint_best.pt")
DQN_PATH = str(MODELS_DIR / "dqn_best.pt")


# ── Temporary flag: set to True to re-enable SAINT + DQN loading ──
LOAD_MODELS = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, init components."""
    print("=" * 60)
    print("  ALSS-LEPC Full Demo — Starting up...")
    print("=" * 60)

    # Init LLM (OpenAI-compatible local endpoint)
    init_llm()

    # Init MongoDB persistence
    print("[0/2] Connecting to MongoDB...")
    await mongo_store.init_mongo()

    if LOAD_MODELS:
        # Create session manager (loads SAINT + DQN models)
        print("[1/2] Loading SAINT + DQN models...")
        manager = SessionManager(
            saint_path=SAINT_PATH,
            dqn_path=DQN_PATH,
        )
        print(f"  SAINT loaded: {manager.n_concepts} concepts")
        print(f"  DQN loaded: ready for action selection")
    else:
        print("[1/2] Model loading DISABLED (LOAD_MODELS=False) — skipping SAINT + DQN")
        manager = None

    # Inject into routers
    session_router.session_manager = manager
    knowledge_router.session_manager = manager
    pipeline_router.session_manager = manager

    print("[2/2] Server ready!")
    print("=" * 60)

    # Store in app state
    app.state.session_manager = manager

    yield

    print("Shutting down...")


app = FastAPI(
    title="ALSS-LEPC Adaptive Learning API",
    description="Adaptive Learning System with SAINT KT + D3QN RL",
    version="1.0.0",
    lifespan=lifespan,
)

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


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/api/info")
async def info():
    manager = app.state.session_manager
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
