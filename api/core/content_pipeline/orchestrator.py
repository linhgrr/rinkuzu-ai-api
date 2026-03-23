"""Compatibility entrypoints for the unified content pipeline."""

from typing import Optional

from ..shared import mongo_store
from .application.pipeline_runner import PipelineRunner
from .application.pipeline_service import PipelineService
from .domain.jobs import PipelineJob
from .infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_SRC,
)

async def process_pdf(
    file_path: str,
    subject_id: Optional[str] = None,
    prs_threshold: float = 0.75,
    min_confidence: float = 0.6,
    apply_reduction: bool = True,
    user_id: Optional[str] = None,
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
    apply_reduction: bool,
):
    await get_pipeline_runner().run(
        job,
        file_path=file_path,
        prs_threshold=prs_threshold,
        min_confidence=min_confidence,
        apply_reduction=apply_reduction,
    )


_pipeline_service: PipelineService | None = None
_pipeline_runner: PipelineRunner | None = None


def get_pipeline_service() -> PipelineService:
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService(
            save_job=mongo_store.save_pipeline_job,
            run_pipeline=_run_pipeline,
        )
    return _pipeline_service


def get_pipeline_runner() -> PipelineRunner:
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = PipelineRunner(
            load_job=mongo_store.load_pipeline_job,
            save_job=mongo_store.save_pipeline_job,
            persist_job_state=get_pipeline_service().persist_job_state,
        )
    return _pipeline_runner
