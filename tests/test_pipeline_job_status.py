"""
tests/test_pipeline_job_status.py

Tests for GET /api/pipeline/jobs/{job_id} — verifies that eta_seconds and
retry_count are present in the response payload.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_current_user
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter
from api.routers import pipeline


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
    from api.rate_limit import limiter as _limiter

    if hasattr(_limiter, "_storage") and hasattr(_limiter._storage, "reset"):
        _limiter._storage.reset()


def _patch_load(monkeypatch, doc):
    async def _fake_load(job_id, user_id):
        return doc

    monkeypatch.setattr(pipeline, "load_pipeline_job_for_user", _fake_load)


def test_get_job_status_returns_404_when_not_found(monkeypatch):
    """GET /jobs/{job_id} returns 404 when job does not exist for the user."""
    _patch_load(monkeypatch, None)
    client = _build_client()

    response = client.get("/api/pipeline/jobs/missing-job")

    assert response.status_code == 404


def test_get_job_status_includes_eta_seconds_and_retry_count(monkeypatch):
    """GET /jobs/{job_id} must include eta_seconds and retry_count in data."""
    _patch_load(
        monkeypatch,
        {
            "job_id": "job-1",
            "status": "extracting",
            "eta_seconds": 90.0,
            "retry_count": 1,
        },
    )
    client = _build_client()

    response = client.get("/api/pipeline/jobs/job-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["eta_seconds"] == 90.0
    assert data["retry_count"] == 1


def test_get_job_status_eta_seconds_defaults_to_none(monkeypatch):
    """When eta_seconds is absent from the doc, the response field must be None."""
    _patch_load(
        monkeypatch,
        {
            "job_id": "job-2",
            "status": "pending",
        },
    )
    client = _build_client()

    response = client.get("/api/pipeline/jobs/job-2")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["eta_seconds"] is None


def test_get_job_status_retry_count_defaults_to_zero(monkeypatch):
    """When retry_count is absent from the doc, the response field must be 0."""
    _patch_load(
        monkeypatch,
        {
            "job_id": "job-3",
            "status": "pending",
        },
    )
    client = _build_client()

    response = client.get("/api/pipeline/jobs/job-3")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["retry_count"] == 0
