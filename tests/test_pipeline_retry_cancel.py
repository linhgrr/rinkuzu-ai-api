from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.core.content_pipeline import PipelineStatus
from api.dependencies import (
    get_content_pipeline_availability,
    get_content_pipeline_service,
    get_current_user,
)
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter
from api.routers import pipeline


class FakePipelineService:
    """Minimal stand-in for PipelineService used by the retry/cancel endpoints."""

    def __init__(self) -> None:
        self.request_cancel_calls: list[Any] = []
        self.retry_job_calls: list[dict[str, Any]] = []
        self.retry_error: Exception | None = None

    def build_job_from_payload(self, doc: dict[str, Any]) -> Any:
        status_value = doc.get("status", PipelineStatus.PENDING.value)
        return SimpleNamespace(
            job_id=doc.get("job_id"),
            status=PipelineStatus(status_value),
            retry_count=doc.get("retry_count", 0),
        )

    async def request_cancel(self, job: Any) -> None:
        self.request_cancel_calls.append(job)

    async def retry_job(self, job: Any, *, download_source: Any, max_retry_count: int) -> None:
        self.retry_job_calls.append(
            {
                "job": job,
                "download_source": download_source,
                "max_retry_count": max_retry_count,
            }
        )
        if self.retry_error is not None:
            raise self.retry_error
        # Mimic a successful retry resetting the job to QUEUED + bumping retry_count.
        job.status = PipelineStatus.QUEUED
        job.retry_count += 1


def _build_client(
    service: FakePipelineService,
    *,
    available: bool = True,
) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.limiter = limiter
    app.include_router(pipeline.router)

    app.dependency_overrides[get_current_user] = lambda: "user-1"
    app.dependency_overrides[get_content_pipeline_service] = lambda: service
    app.dependency_overrides[get_content_pipeline_availability] = lambda: {
        "available": available,
        "error": None,
        "src": "test",
        "service_initialized": True,
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    # Treat every request as an admin call so SlowAPI exempts it from limits.
    monkeypatch.setattr(pipeline, "is_admin_request", lambda *a, **k: True)


def _patch_load(monkeypatch, doc):
    async def _fake_load(job_id, user_id):
        return doc

    monkeypatch.setattr(pipeline, "load_pipeline_job_for_user", _fake_load)


def test_cancel_running_job_requests_cancellation(monkeypatch):
    service = FakePipelineService()
    _patch_load(
        monkeypatch,
        {"job_id": "job-1", "status": PipelineStatus.EXTRACTING.value},
    )
    client = _build_client(service)

    response = client.post("/api/pipeline/jobs/job-1/cancel")

    assert response.status_code == 202
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["job_id"] == "job-1"
    assert payload["data"]["status"] == "cancelling"
    assert len(service.request_cancel_calls) == 1


def test_retry_cancel_openapi_schemas_are_not_untyped_dict():
    """Retry/cancel must export concrete contracts for generated clients."""
    client = _build_client(FakePipelineService())

    schema = client.app.openapi()
    cancel_schema = schema["paths"]["/api/pipeline/jobs/{job_id}/cancel"]["post"]["responses"][
        "202"
    ]["content"]["application/json"]["schema"]
    retry_schema = schema["paths"]["/api/pipeline/jobs/{job_id}/retry"]["post"]["responses"]["202"][
        "content"
    ]["application/json"]["schema"]

    assert cancel_schema["$ref"].endswith("StandardResponse_PipelineJobCancelResponse_")
    assert retry_schema["$ref"].endswith("StandardResponse_PipelineJobRetryResponse_")


def test_cancel_terminal_job_is_noop(monkeypatch):
    service = FakePipelineService()
    _patch_load(
        monkeypatch,
        {"job_id": "job-1", "status": PipelineStatus.COMPLETED.value},
    )
    client = _build_client(service)

    response = client.post("/api/pipeline/jobs/job-1/cancel")

    assert response.status_code in (200, 202)
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == PipelineStatus.COMPLETED.value
    assert payload["meta"]["message"]
    assert service.request_cancel_calls == []


def test_retry_unknown_job_returns_404(monkeypatch):
    service = FakePipelineService()
    _patch_load(monkeypatch, None)
    client = _build_client(service)

    response = client.post("/api/pipeline/jobs/missing/retry")

    assert response.status_code == 404
    assert service.retry_job_calls == []


def test_retry_non_terminal_job_returns_409(monkeypatch):
    service = FakePipelineService()
    _patch_load(
        monkeypatch,
        {"job_id": "job-1", "status": PipelineStatus.EXTRACTING.value, "retryable": True},
    )
    client = _build_client(service)

    response = client.post("/api/pipeline/jobs/job-1/retry")

    assert response.status_code == 409
    assert service.retry_job_calls == []


def test_retry_not_retryable_job_returns_400(monkeypatch):
    service = FakePipelineService()
    _patch_load(
        monkeypatch,
        {"job_id": "job-1", "status": PipelineStatus.FAILED.value, "retryable": False},
    )
    client = _build_client(service)

    response = client.post("/api/pipeline/jobs/job-1/retry")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert service.retry_job_calls == []


def test_retry_terminal_retryable_job_returns_202(monkeypatch):
    service = FakePipelineService()
    _patch_load(
        monkeypatch,
        {
            "job_id": "job-1",
            "status": PipelineStatus.FAILED.value,
            "retryable": True,
            "retry_count": 0,
        },
    )
    client = _build_client(service)

    response = client.post("/api/pipeline/jobs/job-1/retry")

    assert response.status_code == 202
    payload = response.json()
    data = payload["data"]
    assert data["job_id"] == "job-1"
    assert data["status"] == PipelineStatus.QUEUED.value
    assert data["status_url"] == "/api/pipeline/jobs/job-1"
    assert data["retry_count"] == 1

    assert len(service.retry_job_calls) == 1
    call = service.retry_job_calls[0]
    assert call["download_source"] is pipeline.download_source_to_dir
    assert isinstance(call["max_retry_count"], int)


def test_retry_unavailable_pipeline_returns_503(monkeypatch):
    service = FakePipelineService()
    _patch_load(
        monkeypatch,
        {"job_id": "job-1", "status": PipelineStatus.FAILED.value, "retryable": True},
    )
    client = _build_client(service, available=False)

    response = client.post("/api/pipeline/jobs/job-1/retry")

    assert response.status_code == 503
    assert service.retry_job_calls == []
