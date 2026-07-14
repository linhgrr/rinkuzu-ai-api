"""
tests/test_pipeline_job_status.py

Tests for GET /api/v1/pipeline/jobs/{job_id} — verifies that eta_seconds and
retry_count are present in the response payload.
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_current_user, get_session_manager, get_session_service
from api.domains.content_pipeline import router as pipeline
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


def _build_client(*, manager=None, exercise_svc=None) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.limiter = limiter
    app.include_router(pipeline.router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    if manager is not None:
        app.dependency_overrides[get_session_manager] = lambda: manager
    if exercise_svc is not None:
        app.dependency_overrides[get_session_service] = lambda: exercise_svc
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

    async def _fake_status_load(job_id, user_id, *, include_debug=False):
        del job_id, user_id, include_debug
        return doc

    monkeypatch.setattr(pipeline, "load_pipeline_job_for_user", _fake_load)
    monkeypatch.setattr(pipeline, "load_pipeline_job_status_for_user", _fake_status_load)


def _completed_job_doc(job_id: str = "job-1"):
    return {
        "job_id": job_id,
        "status": "completed",
        "result": {
            "concepts_data": {},
            "concept_map": {"c1": 0},
            "prereq_edges": [],
        },
    }


def test_get_job_status_returns_404_when_not_found(monkeypatch):
    """GET /jobs/{job_id} returns 404 when job does not exist for the user."""
    _patch_load(monkeypatch, None)
    client = _build_client()

    response = client.get("/api/v1/pipeline/jobs/missing-job")

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

    response = client.get("/api/v1/pipeline/jobs/job-1")

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

    response = client.get("/api/v1/pipeline/jobs/job-2")

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

    response = client.get("/api/v1/pipeline/jobs/job-3")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["retry_count"] == 0


def test_get_job_status_uses_compact_projection_by_default(monkeypatch):
    calls = []

    async def _fake_status_load(job_id, user_id, *, include_debug=False):
        calls.append((job_id, user_id, include_debug))
        return {
            "job_id": job_id,
            "status": "completed",
        }

    monkeypatch.setattr(pipeline, "load_pipeline_job_status_for_user", _fake_status_load)
    client = _build_client()

    response = client.get("/api/v1/pipeline/jobs/job-compact")

    assert response.status_code == 200
    assert calls == [("job-compact", "user-1", False)]
    assert response.json()["data"]["result"] is None
    assert response.json()["data"]["debug_trace"] == []


def test_get_job_status_includes_debug_payload_only_when_requested(monkeypatch):
    calls = []

    async def _fake_status_load(job_id, user_id, *, include_debug=False):
        calls.append((job_id, user_id, include_debug))
        return {
            "job_id": job_id,
            "status": "completed",
            "debug_trace": [],
            "result": {
                "concept_map": {"c1": 0},
                "concepts_data": {},
                "prereq_edges": [],
                "concept_embedding_count": 23,
            },
        }

    monkeypatch.setattr(pipeline, "load_pipeline_job_status_for_user", _fake_status_load)
    client = _build_client()

    response = client.get("/api/v1/pipeline/jobs/job-debug?include_debug=true")

    assert response.status_code == 200
    assert calls == [("job-debug", "user-1", True)]
    assert response.json()["data"]["result"]["concept_embedding_count"] == 23


def test_create_session_reuses_existing_active_session(monkeypatch):
    """POST /create-session should be idempotent for an active saved subject session."""
    _patch_load(monkeypatch, _completed_job_doc())

    async def _fake_load_subject_progress(job_id, user_id):
        assert (job_id, user_id) == ("job-1", "user-1")
        return {"last_session_id": "session-1"}

    monkeypatch.setattr(pipeline, "load_subject_progress_for_user", _fake_load_subject_progress)

    class FakeManager:
        def __init__(self):
            self.create_calls = 0

        def get_active_pipeline_session(self, user_id, job_id):
            assert (user_id, job_id) == ("user-1", "job-1")

        async def get_or_create_pipeline_session(self, **kwargs):
            assert kwargs["job_doc"]["job_id"] == "job-1"
            assert kwargs["subject_progress"]["last_session_id"] == "session-1"
            assert kwargs["user_id"] == "user-1"
            self.create_calls += 1
            return (
                SimpleNamespace(session_id="session-1", job_id="job-1", status="active"),
                False,
            )

    class FakeExerciseService:
        def __init__(self):
            self.prefetch_calls = 0

        async def eager_generate_first_exercise(self, session):
            del session
            self.prefetch_calls += 1

    manager = FakeManager()
    exercise_svc = FakeExerciseService()
    client = _build_client(manager=manager, exercise_svc=exercise_svc)

    response = client.post("/api/v1/pipeline/jobs/job-1/create-session")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["session_id"] == "session-1"
    assert data["source"] == "existing_session"
    assert manager.create_calls == 1
    assert exercise_svc.prefetch_calls == 0


def test_create_session_active_fast_path_skips_persistence_reads(monkeypatch):
    async def _unexpected_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("active session fast path should not read persistence")

    monkeypatch.setattr(pipeline, "load_pipeline_job_for_user", _unexpected_load)
    monkeypatch.setattr(pipeline, "load_subject_progress_for_user", _unexpected_load)

    class FakeManager:
        def get_active_pipeline_session(self, user_id, job_id):
            assert (user_id, job_id) == ("user-1", "job-1")
            return SimpleNamespace(
                session_id="session-1",
                concept_map={"c1": 0},
                status="active",
            )

    client = _build_client(manager=FakeManager(), exercise_svc=SimpleNamespace())
    response = client.post("/api/v1/pipeline/jobs/job-1/create-session")

    assert response.status_code == 200
    assert response.json()["data"]["source"] == "existing_session"
