"""Cache-hit chunk rebuild must not leave a false COMPLETED usable state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner
from api.domains.content_pipeline.application.stages.cache_restore import S3CacheRestoreResult
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.shared.persistence.pipeline_jobs import SaveJobOutcome


@pytest.mark.asyncio
async def test_rebuild_failure_persists_terminal_failure_not_completed(monkeypatch):
    job = PipelineJob(job_id="j-cache", filename="a.pdf", subject_id="s1")
    job.result = {"concept_map": {"c1": 0}}
    job.current_step = "Loaded from S3 cache"
    job.status = PipelineStatus.LOADING

    saved: list[PipelineJob] = []

    async def save_job(j: PipelineJob) -> bool:
        saved.append(
            SimpleNamespace(
                status=j.status,
                error_code=j.error_code,
                retryable=j.retryable,
            )
        )
        return SaveJobOutcome.APPLIED

    async def persist_job_state(j, status, step, progress):
        j.status = status
        j.current_step = step
        j.progress = progress
        await save_job(j)

    runner = PipelineRunner(
        load_job=AsyncMock(return_value=None),
        load_cancel_flag=AsyncMock(return_value=False),
        save_job=save_job,
        persist_job_state=persist_job_state,
        chunk_chroma_store=None,
    )

    monkeypatch.setattr(
        "api.domains.content_pipeline.application.pipeline_runner.get_content_processor_bindings",
        dict,
    )
    monkeypatch.setattr(
        "api.domains.content_pipeline.application.pipeline_runner.try_restore_completed_job_from_mongo",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "api.domains.content_pipeline.application.pipeline_runner.get_s3_client",
        object,
    )

    async def _fake_s3_restore(*_a, **_k):
        job.result = {"concept_map": {"c1": 0}}
        job.current_step = "Loaded from S3 cache"
        job.status = PipelineStatus.LOADING
        return S3CacheRestoreResult(cache_key="cache/key.json", restored=True)

    monkeypatch.setattr(
        "api.domains.content_pipeline.application.pipeline_runner.try_restore_completed_job_from_s3",
        _fake_s3_restore,
    )

    async def _boom(*_a, **_k):
        raise RuntimeError("chunk rebuild failed")

    monkeypatch.setattr(runner, "_rebuild_reusable_chunks", _boom)
    monkeypatch.setattr(
        "api.domains.content_pipeline.application.pipeline_runner._resolve_effective_job_timeout",
        AsyncMock(return_value=30.0),
    )

    await runner.run(
        job,
        file_path="/tmp/a.pdf",
        prs_threshold=None,
        min_confidence=0.6,
        apply_reduction=True,
        page_batch_size=10,
    )

    assert job.status is not PipelineStatus.COMPLETED
    assert job.status is PipelineStatus.FAILED
    assert job.retryable is True
    assert job.error_code == "pipeline_cache_rebuild_failed"
    assert any(getattr(s, "status", None) is not PipelineStatus.COMPLETED for s in saved)
