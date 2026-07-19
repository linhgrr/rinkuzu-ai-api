"""Tests for cooperative job cancellation primitives and runner integration."""

from __future__ import annotations

import asyncio

import pytest

from api.domains.content_pipeline.application.cancellation import (
    JobCancelledError,
    raise_if_cancelled,
)
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.shared.persistence.pipeline_jobs import SaveJobOutcome

# ---------------------------------------------------------------------------
# PART A — raise_if_cancelled primitive
# ---------------------------------------------------------------------------


def test_raise_if_cancelled_noop_when_not_requested():
    job = PipelineJob(job_id="j", filename="a.pdf", subject_id="a")
    raise_if_cancelled(job)


def test_raise_if_cancelled_raises_when_requested():
    job = PipelineJob(job_id="j", filename="a.pdf", subject_id="a")
    job.cancel_requested = True
    with pytest.raises(JobCancelledError):
        raise_if_cancelled(job)


# ---------------------------------------------------------------------------
# PART C — _check_cancelled helper (unit-testable slice)
# ---------------------------------------------------------------------------


def test_check_cancelled_sets_flag_and_raises_when_doc_has_cancel_requested():
    """_check_cancelled should read the cancel flag and raise if cancel_requested."""
    from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner

    saved: list[PipelineJob] = []

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return True

    async def save_job(job: PipelineJob):
        saved.append(job)
        return SaveJobOutcome.APPLIED

    async def persist_job_state(job, status, step, progress):
        return None

    runner = PipelineRunner(
        load_job=load_job,
        load_cancel_flag=load_cancel_flag,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )

    job = PipelineJob(job_id="j", filename="a.pdf", subject_id="a")

    with pytest.raises(JobCancelledError):
        asyncio.run(runner._check_cancelled(job))

    # The in-memory flag should be updated too
    assert job.cancel_requested is True


def test_check_cancelled_noop_when_doc_has_no_cancel_flag():
    """_check_cancelled should be silent when cancel_requested is False."""
    from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return False

    async def save_job(job: PipelineJob):
        return SaveJobOutcome.APPLIED

    async def persist_job_state(job, status, step, progress):
        return None

    runner = PipelineRunner(
        load_job=load_job,
        load_cancel_flag=load_cancel_flag,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )

    job = PipelineJob(job_id="j", filename="a.pdf", subject_id="a")
    asyncio.run(runner._check_cancelled(job))  # must not raise


def test_check_cancelled_noop_when_doc_missing():
    """_check_cancelled should be silent when the cancel flag read returns False."""
    from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return False

    async def save_job(job: PipelineJob):
        return SaveJobOutcome.APPLIED

    async def persist_job_state(job, status, step, progress):
        return None

    runner = PipelineRunner(
        load_job=load_job,
        load_cancel_flag=load_cancel_flag,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )

    job = PipelineJob(job_id="j", filename="a.pdf", subject_id="a")
    asyncio.run(runner._check_cancelled(job))  # must not raise


# ---------------------------------------------------------------------------
# PART C — _persist_cancelled helper produces CANCELLED status
# ---------------------------------------------------------------------------


def test_persist_cancelled_sets_cancelled_state():
    """_persist_cancelled must persist CANCELLED status, right error_code, retryable=True."""
    from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner

    saved: list[PipelineJob] = []

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return False

    async def save_job(job: PipelineJob):
        saved.append(job)
        return SaveJobOutcome.APPLIED

    async def persist_job_state(job, status, step, progress):
        return None

    runner = PipelineRunner(
        load_job=load_job,
        load_cancel_flag=load_cancel_flag,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )

    job = PipelineJob(job_id="j2", filename="b.pdf", subject_id="b")
    exc = JobCancelledError("Job j2 cancelled by user")

    asyncio.run(runner._persist_cancelled(job, exc))

    assert len(saved) == 1
    persisted = saved[0]
    assert persisted.status is PipelineStatus.CANCELLED
    assert persisted.error_code == "pipeline_cancelled"
    assert persisted.retryable is True
    assert persisted.user_message == "Processing was cancelled."


def test_persist_cancelled_stale_generation_stops_cleanly():
    from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner
    from api.domains.content_pipeline.domain.errors import PipelineStaleWorkerError

    async def load_job(_job_id: str):
        return None

    async def load_cancel_flag(_job_id: str):
        return False

    async def save_job(_job: PipelineJob):
        return SaveJobOutcome.STALE_GENERATION

    async def persist_job_state(_job, _status, _step, _progress):
        return None

    runner = PipelineRunner(
        load_job=load_job,
        load_cancel_flag=load_cancel_flag,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )
    job = PipelineJob(job_id="j3", filename="c.pdf", subject_id="c")

    with pytest.raises(PipelineStaleWorkerError):
        asyncio.run(runner._persist_cancelled(job, JobCancelledError("cancelled")))


@pytest.mark.asyncio
async def test_final_checkpoint_cancel_race_persists_cancelled_not_completed():
    """Race: final checkpoint passes in-memory, save of COMPLETED sees cancel flag."""
    from types import SimpleNamespace

    from api.domains.content_pipeline.application.pipeline_runner import PipelineRunner
    from api.domains.content_pipeline.application.ports import raise_for_save_outcome
    from api.domains.content_pipeline.application.stages import finalization as finalization_stage

    saved: list[SimpleNamespace] = []

    async def load_job(_job_id: str):
        return None

    async def load_cancel_flag(_job_id: str):
        return False  # checkpoint misses the race window

    async def save_job(job: PipelineJob):
        saved.append(SimpleNamespace(status=job.status, error_code=job.error_code))
        # First durable COMPLETED attempt loses to cancel.
        if job.status is PipelineStatus.COMPLETED:
            return SaveJobOutcome.CANCEL_REQUESTED
        return SaveJobOutcome.APPLIED

    async def persist_job_state(job, status, step, progress):
        job.status = status
        job.current_step = step
        job.progress = progress
        raise_for_save_outcome(job, await save_job(job), operation="persist")

    runner = PipelineRunner(
        load_job=load_job,
        load_cancel_flag=load_cancel_flag,
        save_job=save_job,
        persist_job_state=persist_job_state,
    )

    job = PipelineJob(job_id="race-1", filename="a.pdf", subject_id="s1")
    job.result = {"concept_map": {}}

    # Drive finalization as the runner does after the last checkpoint.
    with pytest.raises(JobCancelledError):
        await finalization_stage.complete_pipeline_job(
            job,
            persist_job_state=persist_job_state,
        )

    # Cooperative cancel path after CANCEL_REQUESTED.
    await runner._persist_cancelled(job, JobCancelledError("cancel wins"))
    assert job.status is PipelineStatus.CANCELLED
    assert job.error_code == "pipeline_cancelled"
    assert any(s.status is PipelineStatus.CANCELLED for s in saved)
    # COMPLETED may be attempted once, but must lose to CANCEL_REQUESTED (not applied).
    assert saved[-1].status is PipelineStatus.CANCELLED
