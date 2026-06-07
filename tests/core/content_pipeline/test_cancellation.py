"""Tests for cooperative job cancellation primitives and runner integration."""

from __future__ import annotations

import asyncio

import pytest

from api.core.content_pipeline.application.cancellation import (
    JobCancelledError,
    raise_if_cancelled,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

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
    from api.core.content_pipeline.application.pipeline_runner import PipelineRunner

    saved: list[PipelineJob] = []

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return True

    async def save_job(job: PipelineJob):
        saved.append(job)
        return True

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
    from api.core.content_pipeline.application.pipeline_runner import PipelineRunner

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return False

    async def save_job(job: PipelineJob):
        return True

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
    from api.core.content_pipeline.application.pipeline_runner import PipelineRunner

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return False

    async def save_job(job: PipelineJob):
        return True

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
    from api.core.content_pipeline.application.pipeline_runner import PipelineRunner

    saved: list[PipelineJob] = []

    async def load_job(job_id: str):
        return None

    async def load_cancel_flag(job_id: str):
        return False

    async def save_job(job: PipelineJob):
        saved.append(job)
        return True

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
