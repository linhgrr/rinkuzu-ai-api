"""
tests/test_pipeline_list_jobs.py

Tests for GET /api/pipeline/jobs — user-scoped listing of recent pipeline
jobs of all statuses, with per-job live fields (is_terminal, is_delayed,
retry_after_seconds, progress, retryable, eta_seconds).
"""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_current_user
from api.domains.content_pipeline import router as pipeline
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = time.time()

_PROCESSING_JOB: dict = {
    "job_id": "job-proc-1",
    "filename": "notes.pdf",
    "subject_id": "sub-1",
    "status": "extracting",
    "current_step": "Extracting concepts",
    "progress": 0.3,
    "page_batch_size": 10,
    "batch_count": 3,
    "failed_batch_count": 0,
    "partial_success": False,
    "concepts_extracted": 10,
    "concepts_after_merge": 8,
    "relations_verified": 5,
    "error_code": None,
    "user_message": None,
    "retryable": False,
    "retry_count": 0,
    "eta_seconds": 120,
    "created_at": _NOW - 300,
    "updated_at": _NOW - 5,
    "heartbeat_at": _NOW - 5,  # recent heartbeat → not delayed
    "completed_at": None,
}

_FAILED_JOB: dict = {
    "job_id": "job-fail-1",
    "filename": "slides.pdf",
    "subject_id": "sub-2",
    "status": "failed",
    "current_step": "Failed",
    "progress": 0.0,
    "page_batch_size": 10,
    "batch_count": 2,
    "failed_batch_count": 2,
    "partial_success": False,
    "concepts_extracted": 0,
    "concepts_after_merge": 0,
    "relations_verified": 0,
    "error_code": "pipeline_failed",
    "user_message": "Processing failed",
    "retryable": True,
    "retry_count": 0,
    "eta_seconds": None,
    "created_at": _NOW - 600,
    "updated_at": _NOW - 100,
    "heartbeat_at": _NOW - 100,
    "completed_at": _NOW - 100,
}

_FAKE_JOBS = [_PROCESSING_JOB, _FAILED_JOB]


def _build_client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.limiter = limiter
    app.include_router(pipeline.router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """Treat every request as admin so SlowAPI never rate-limits tests.

    Also resets the in-memory limiter storage so accumulated counts from
    previous tests in the session don't cause spurious 429 responses.
    """
    monkeypatch.setattr(pipeline, "is_admin_request", lambda *a, **k: True)
    # Reset SlowAPI's in-memory counter store so previous tests don't cause
    # spurious 429 responses when the shared limiter accumulates counts.
    from api.rate_limit import limiter as _limiter

    if hasattr(_limiter, "_storage") and hasattr(_limiter._storage, "reset"):
        _limiter._storage.reset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_jobs_returns_200_with_jobs(monkeypatch):
    """GET /api/pipeline/jobs returns 200 with data.jobs and data.count."""

    async def _fake_list(*, user_id: str, limit: int) -> list:
        return list(_FAKE_JOBS)

    monkeypatch.setattr(pipeline, "list_recent_pipeline_jobs_all_status", _fake_list)
    client = _build_client()

    response = client.get("/api/pipeline/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert len(data["jobs"]) == 2
    assert data["count"] == 2


def test_list_jobs_openapi_schema_is_not_untyped_dict():
    """GET /api/pipeline/jobs must export a concrete contract for generated clients."""
    client = _build_client()

    schema = client.app.openapi()
    response_schema = schema["paths"]["/api/pipeline/jobs"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert response_schema["$ref"].endswith("StandardResponse_PipelineJobListResponse_")


def test_list_jobs_each_job_has_live_fields(monkeypatch):
    """Every job in the listing must include is_terminal, is_delayed, retry_after_seconds."""

    async def _fake_list(*, user_id: str, limit: int) -> list:
        return list(_FAKE_JOBS)

    monkeypatch.setattr(pipeline, "list_recent_pipeline_jobs_all_status", _fake_list)
    client = _build_client()

    response = client.get("/api/pipeline/jobs")

    assert response.status_code == 200
    jobs = response.json()["data"]["jobs"]
    for job in jobs:
        assert "is_terminal" in job, f"job {job.get('job_id')} missing is_terminal"
        assert "is_delayed" in job, f"job {job.get('job_id')} missing is_delayed"
        assert "retry_after_seconds" in job, f"job {job.get('job_id')} missing retry_after_seconds"
        # Passthrough fields must also be present
        assert "status" in job
        assert "progress" in job
        assert "retryable" in job
        assert "eta_seconds" in job


def test_list_jobs_failed_job_is_terminal(monkeypatch):
    """The failed job must have is_terminal=True and retry_after_seconds=0."""

    async def _fake_list(*, user_id: str, limit: int) -> list:
        return list(_FAKE_JOBS)

    monkeypatch.setattr(pipeline, "list_recent_pipeline_jobs_all_status", _fake_list)
    client = _build_client()

    response = client.get("/api/pipeline/jobs")

    jobs = {j["job_id"]: j for j in response.json()["data"]["jobs"]}
    failed = jobs["job-fail-1"]
    assert failed["is_terminal"] is True
    assert failed["retry_after_seconds"] == 0


def test_list_jobs_processing_job_is_not_terminal(monkeypatch):
    """The extracting job must have is_terminal=False."""

    async def _fake_list(*, user_id: str, limit: int) -> list:
        return list(_FAKE_JOBS)

    monkeypatch.setattr(pipeline, "list_recent_pipeline_jobs_all_status", _fake_list)
    client = _build_client()

    response = client.get("/api/pipeline/jobs")

    jobs = {j["job_id"]: j for j in response.json()["data"]["jobs"]}
    proc = jobs["job-proc-1"]
    assert proc["is_terminal"] is False


def test_list_jobs_empty_returns_zero_count(monkeypatch):
    """When user has no jobs, data.count must be 0 and data.jobs must be []."""

    async def _fake_list(*, user_id: str, limit: int) -> list:
        return []

    monkeypatch.setattr(pipeline, "list_recent_pipeline_jobs_all_status", _fake_list)
    client = _build_client()

    response = client.get("/api/pipeline/jobs")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["jobs"] == []
    assert data["count"] == 0


def test_list_jobs_custom_limit_is_forwarded(monkeypatch):
    """The limit query param must be forwarded to the persistence layer."""
    received: list[int] = []

    async def _fake_list(*, user_id: str, limit: int) -> list:
        received.append(limit)
        return []

    monkeypatch.setattr(pipeline, "list_recent_pipeline_jobs_all_status", _fake_list)
    client = _build_client()

    client.get("/api/pipeline/jobs?limit=10")

    assert received == [10]
