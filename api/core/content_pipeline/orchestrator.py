"""Compatibility entrypoints for the unified content pipeline."""

import functools

from api.core.shared import mongo_store

from .application.pipeline_runner import PipelineRunner
from .application.pipeline_service import PipelineService
from .domain.jobs import PipelineJob
from .infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_SRC,
)


async def process_pdf(
    file_path: str,
    subject_id: str | None = None,
    prs_threshold: float = 0.75,
    min_confidence: float = 0.6,
    *,
    apply_reduction: bool = True,
    user_id: str | None = None,
) -> PipelineJob:
    return await get_pipeline_service().start_job(
        file_path=file_path,
        subject_id=subject_id,
        prs_threshold=prs_threshold,
        min_confidence=min_confidence,
        apply_reduction=apply_reduction,
        user_id=user_id,
        content_processor_available=CONTENT_PROCESSOR_AVAILABLE,
        content_processor_src=CONTENT_PROCESSOR_SRC,
    )


async def _run_pipeline(
    job: PipelineJob,
    file_path: str,
    prs_threshold: float,
    min_confidence: float,
    *,
    apply_reduction: bool,
):
    await get_pipeline_runner().run(
        job,
        file_path=file_path,
        prs_threshold=prs_threshold,
        min_confidence=min_confidence,
        apply_reduction=apply_reduction,
    )


@functools.lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    return PipelineService(
        save_job=mongo_store.save_pipeline_job,
        run_pipeline=_run_pipeline,
    )


@functools.lru_cache(maxsize=1)
def get_pipeline_runner() -> PipelineRunner:
    return PipelineRunner(
        load_job=mongo_store.load_pipeline_job,
        save_job=mongo_store.save_pipeline_job,
        persist_job_state=get_pipeline_service().persist_job_state,
    )
