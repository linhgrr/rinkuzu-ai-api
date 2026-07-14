"""Application startup/shutdown wiring for the FastAPI app.

Holds the ``lifespan`` context manager and its component initializers, kept
out of main.py so that module stays a thin app factory.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import time
from typing import Any

from fastapi import FastAPI
from loguru import logger

from .config import Settings, get_settings
from .domains.content_pipeline.application.pipeline_runner import PipelineRunner
from .domains.content_pipeline.application.pipeline_service import PipelineService
from .domains.content_pipeline.application.recovery import PipelineJanitor
from .domains.content_pipeline.application.source_fetch import download_source_to_dir
from .domains.content_pipeline.application.stages.execution import shutdown_pipeline_executor
from .domains.content_pipeline.application.stages.model_worker import (
    shutdown_sentence_transformer_worker,
)
from .domains.content_pipeline.domain.jobs import PipelineJob
from .domains.content_pipeline.infrastructure.embed.embedding_client import EmbeddingClient
from .domains.content_pipeline.infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_ERROR,
    CONTENT_PROCESSOR_SRC,
)
from .domains.content_pipeline.infrastructure.storage.chunk_chroma_store import ChunkChromaStore
from .domains.learning.exercise_service import ExerciseService
from .domains.learning.session import SessionManager
from .domains.quiz.draft_tasks import quiz_draft_task_manager
from .observability import setup_otel, shutdown_otel
from .shared import mongo_store
from .shared.persistence import (
    load_pipeline_job,
    load_pipeline_job_cancel_requested,
    save_pipeline_job,
)
from .shared.persistence.pipeline_jobs import list_active_pipeline_jobs

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
    if not get_settings().load_models:
        app.state.chunk_chroma_store = None
        logger.info("[RAG] Model loading disabled — skipping ChunkChromaStore")
        return

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
    from .shared.llm import normalize_llm_base_url

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
    if mongo_store.is_available():
        await quiz_draft_task_manager.recover()

    janitor = getattr(app.state, "pipeline_janitor", None)
    if janitor is not None and CONTENT_PROCESSOR_AVAILABLE and mongo_store.is_available():
        await janitor.start()

    logger.info("[2/2] Server ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")
    await quiz_draft_task_manager.shutdown()
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
