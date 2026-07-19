from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import (
    get_content_pipeline_availability,
    get_content_pipeline_service,
    get_current_user,
)
from api.domains.content_pipeline import PipelineStatus
from api.domains.content_pipeline import router as pipeline
from api.domains.content_pipeline.application.cancellation import JobCancelledError
from api.domains.content_pipeline.domain.errors import (
    PipelineSchedulingUnavailableError,
    PipelineSourceDownloadError,
)
from api.domains.content_pipeline.domain.transitions import (
    RetryCompensationOutcome,
    RetryCompensationResult,
)
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


class FakePipelineService:
    """Minimal stand-in for PipelineService used by the retry/cancel endpoints."""

    def __init__(self) -> None:
        self.request_cancel_calls: list[Any] = []
        self.retry_job_calls: list[dict[str, Any]] = []
        self.reschedule_calls: list[dict[str, Any]] = []
        self.retry_error: Exception | None = None

    def build_job_from_payload(self, doc: dict[str, Any]) -> Any:
        status_value = doc.get("status", PipelineStatus.PENDING.value)
        return SimpleNamespace(
            job_id=doc.get("job_id"),
            status=PipelineStatus(status_value),
            retry_count=doc.get("retry_count", 0),
            source_s3_key=doc.get("source_s3_key", "uploads/x.pdf"),
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
        job.status = PipelineStatus.QUEUED
        job.retry_count += 1

    async def reschedule_retried_job(self, job: Any, *, download_source: Any) -> None:
        self.reschedule_calls.append({"job": job, "download_source": download_source})
        if self.retry_error is not None:
            raise self.retry_error


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
def _no_rate_limit():
    """Disable the shared limiter for this module and restore it after each test."""
    was_enabled = limiter.enabled
    limiter.enabled = False
    try:
        yield
    finally:
        limiter.enabled = was_enabled


def _patch_load(monkeypatch, doc):
    async def _fake_load(job_id, user_id):
        return doc

    monkeypatch.setattr(pipeline, "load_pipeline_job_for_user", _fake_load)


def _patch_cancel(monkeypatch, result):
    async def _fake_cancel(job_id, user_id):
        return result

    monkeypatch.setattr(pipeline, "request_cancel_pipeline_job_for_user", _fake_cancel)


def _patch_retry_transition(monkeypatch, result):
    async def _fake_transition(job_id, user_id, *, max_retry_count):
        return result

    monkeypatch.setattr(pipeline, "transition_pipeline_job_for_retry", _fake_transition)


def test_cancel_running_job_requests_cancellation(monkeypatch):
    from api.shared.persistence.pipeline_jobs import CancelJobOutcome, CancelJobResult

    service = FakePipelineService()
    _patch_cancel(
        monkeypatch,
        CancelJobResult(
            outcome=CancelJobOutcome.REQUESTED,
            status=PipelineStatus.EXTRACTING.value,
            cancel_requested=True,
        ),
    )
    client = _build_client(service)

    response = client.post("/api/v1/pipeline/jobs/job-1/cancel")

    assert response.status_code == 202
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["job_id"] == "job-1"
    assert payload["data"]["status"] == "cancelling"


def test_retry_cancel_openapi_schemas_are_not_untyped_dict():
    """Retry/cancel must export concrete contracts for generated clients."""
    client = _build_client(FakePipelineService())

    schema = client.app.openapi()
    cancel_schema = schema["paths"]["/api/v1/pipeline/jobs/{job_id}/cancel"]["post"]["responses"][
        "202"
    ]["content"]["application/json"]["schema"]
    retry_schema = schema["paths"]["/api/v1/pipeline/jobs/{job_id}/retry"]["post"]["responses"][
        "202"
    ]["content"]["application/json"]["schema"]

    assert cancel_schema["$ref"].endswith("StandardResponse_PipelineJobCancelResponse_")
    assert retry_schema["$ref"].endswith("StandardResponse_PipelineJobRetryResponse_")


def test_cancel_terminal_job_is_noop(monkeypatch):
    from api.shared.persistence.pipeline_jobs import CancelJobOutcome, CancelJobResult

    service = FakePipelineService()
    _patch_cancel(
        monkeypatch,
        CancelJobResult(
            outcome=CancelJobOutcome.ALREADY_TERMINAL,
            status=PipelineStatus.COMPLETED.value,
        ),
    )
    client = _build_client(service)

    response = client.post("/api/v1/pipeline/jobs/job-1/cancel")

    assert response.status_code in (200, 202)
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == PipelineStatus.COMPLETED.value
    assert payload["meta"]["message"]
    assert service.request_cancel_calls == []


def test_cancel_conflict_returns_retryable_409(monkeypatch):
    from api.shared.persistence.pipeline_jobs import CancelJobOutcome, CancelJobResult

    service = FakePipelineService()
    _patch_cancel(
        monkeypatch,
        CancelJobResult(
            outcome=CancelJobOutcome.CONFLICT,
            status=PipelineStatus.QUEUED.value,
        ),
    )
    response = _build_client(service).post("/api/v1/pipeline/jobs/job-1/cancel")

    assert response.status_code == 409
    assert response.json()["error"]["meta"]["retryable"] is True


def test_cancel_does_not_require_pipeline_runtime_service(monkeypatch):
    from api.shared.persistence.pipeline_jobs import CancelJobOutcome, CancelJobResult

    service = FakePipelineService()
    _patch_cancel(
        monkeypatch,
        CancelJobResult(
            outcome=CancelJobOutcome.REQUESTED,
            status=PipelineStatus.EXTRACTING.value,
            cancel_requested=True,
        ),
    )
    client = _build_client(service)

    def _runtime_unavailable():
        raise AssertionError("cancel must not resolve PipelineService")

    client.app.dependency_overrides[get_content_pipeline_service] = _runtime_unavailable
    response = client.post("/api/v1/pipeline/jobs/job-1/cancel")
    assert response.status_code == 202


def test_retry_unknown_job_returns_404(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    _patch_retry_transition(monkeypatch, RetryJobResult(outcome=RetryJobOutcome.NOT_FOUND))
    client = _build_client(service)

    response = client.post("/api/v1/pipeline/jobs/missing/retry")

    assert response.status_code == 404
    assert service.reschedule_calls == []


def test_retry_non_terminal_job_returns_409(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(outcome=RetryJobOutcome.INVALID_STATE),
    )
    client = _build_client(service)

    response = client.post("/api/v1/pipeline/jobs/job-1/retry")

    assert response.status_code == 409
    assert service.reschedule_calls == []


def test_retry_not_retryable_job_returns_400(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(outcome=RetryJobOutcome.NOT_RETRYABLE),
    )
    client = _build_client(service)

    response = client.post("/api/v1/pipeline/jobs/job-1/retry")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert service.reschedule_calls == []


def test_retry_terminal_retryable_job_returns_202(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(
            outcome=RetryJobOutcome.RETRIED,
            job={
                "job_id": "job-1",
                "filename": "a.pdf",
                "subject_id": "s1",
                "status": PipelineStatus.QUEUED.value,
                "retryable": False,
                "retry_count": 1,
                "source_s3_key": "uploads/x.pdf",
                "cancel_requested": False,
            },
        ),
    )
    client = _build_client(service)

    response = client.post("/api/v1/pipeline/jobs/job-1/retry")

    assert response.status_code == 202
    payload = response.json()
    data = payload["data"]
    assert data["job_id"] == "job-1"
    assert data["status"] == PipelineStatus.QUEUED.value
    assert data["status_url"] == "/api/v1/pipeline/jobs/job-1"
    assert data["retry_count"] == 1

    assert len(service.reschedule_calls) == 1
    call = service.reschedule_calls[0]
    assert call["download_source"] is pipeline.download_source_to_dir


def test_retry_unavailable_pipeline_returns_503(monkeypatch):
    service = FakePipelineService()
    client = _build_client(service, available=False)

    response = client.post("/api/v1/pipeline/jobs/job-1/retry")

    assert response.status_code == 503
    assert service.reschedule_calls == []


def test_cancel_storage_outage_returns_503(monkeypatch):
    from pymongo.errors import ServerSelectionTimeoutError

    service = FakePipelineService()

    async def _boom(job_id, user_id):
        raise ServerSelectionTimeoutError("mongo down")

    monkeypatch.setattr(pipeline, "request_cancel_pipeline_job_for_user", _boom)
    client = _build_client(service)
    response = client.post("/api/v1/pipeline/jobs/job-1/cancel")
    assert response.status_code == 503
    assert response.json()["error"]["meta"]["retryable"] is True


def test_double_cancel_endpoint_is_idempotent(monkeypatch):
    from api.shared.persistence.pipeline_jobs import CancelJobOutcome, CancelJobResult

    service = FakePipelineService()
    _patch_cancel(
        monkeypatch,
        CancelJobResult(
            outcome=CancelJobOutcome.REQUESTED,
            status=PipelineStatus.EXTRACTING.value,
            cancel_requested=True,
        ),
    )
    client = _build_client(service)
    first = client.post("/api/v1/pipeline/jobs/job-1/cancel")
    second = client.post("/api/v1/pipeline/jobs/job-1/cancel")
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["data"]["status"] == "cancelling"
    assert second.json()["data"]["status"] == "cancelling"


def test_retry_download_failure_compensates_and_returns_503(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    service.retry_error = PipelineSourceDownloadError("download failed")
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(
            outcome=RetryJobOutcome.RETRIED,
            job={
                "job_id": "job-1",
                "filename": "a.pdf",
                "subject_id": "s1",
                "status": PipelineStatus.QUEUED.value,
                "retryable": False,
                "retry_count": 1,
                "source_s3_key": "uploads/x.pdf",
                "cancel_requested": False,
            },
        ),
    )
    compensate_calls: list[dict] = []

    async def _fake_compensate(job_id, user_id, *, retry_count, retryable):
        compensate_calls.append(
            {
                "job_id": job_id,
                "user_id": user_id,
                "retry_count": retry_count,
                "retryable": retryable,
            }
        )
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.APPLIED,
            status=PipelineStatus.FAILED.value,
            retry_count=retry_count,
        )

    monkeypatch.setattr(pipeline, "compensate_failed_retry_reschedule", _fake_compensate)
    client = _build_client(service)
    response = client.post("/api/v1/pipeline/jobs/job-1/retry")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["meta"]["retryable"] is True
    assert body["error"]["meta"]["error_code"] == "pipeline_retry_reschedule_failed"
    assert compensate_calls == [
        {"job_id": "job-1", "user_id": "user-1", "retry_count": 1, "retryable": True}
    ]
    assert len(service.reschedule_calls) == 1


def test_retry_shutdown_failure_compensates_and_returns_503(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    service.retry_error = PipelineSchedulingUnavailableError(
        "Content pipeline is shutting down and cannot reschedule jobs."
    )
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(
            outcome=RetryJobOutcome.RETRIED,
            job={
                "job_id": "job-1",
                "filename": "a.pdf",
                "subject_id": "s1",
                "status": PipelineStatus.QUEUED.value,
                "retryable": False,
                "retry_count": 2,
                "source_s3_key": "uploads/x.pdf",
                "cancel_requested": False,
            },
        ),
    )

    async def _fake_compensate(job_id, user_id, *, retry_count, retryable):
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.APPLIED,
            status=PipelineStatus.FAILED.value,
            retry_count=retry_count,
        )

    monkeypatch.setattr(pipeline, "compensate_failed_retry_reschedule", _fake_compensate)
    response = _build_client(service).post("/api/v1/pipeline/jobs/job-1/retry")
    assert response.status_code == 503
    assert response.json()["error"]["meta"]["retryable"] is True


def test_retry_programming_error_is_500_after_nonretryable_compensation(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    service.retry_error = ValueError("broken invariant")
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(
            outcome=RetryJobOutcome.RETRIED,
            job={
                "job_id": "job-1",
                "filename": "a.pdf",
                "subject_id": "s1",
                "status": PipelineStatus.QUEUED.value,
                "retry_count": 1,
                "source_s3_key": "uploads/x.pdf",
            },
        ),
    )
    compensation_flags: list[bool] = []

    async def _fake_compensate(job_id, user_id, *, retry_count, retryable):
        compensation_flags.append(retryable)
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.APPLIED,
            status=PipelineStatus.FAILED.value,
            retry_count=retry_count,
        )

    monkeypatch.setattr(pipeline, "compensate_failed_retry_reschedule", _fake_compensate)
    response = _build_client(service).post("/api/v1/pipeline/jobs/job-1/retry")
    assert response.status_code == 500
    assert compensation_flags == [False]


def test_retry_cancel_during_download_preserves_cancel_and_returns_409(monkeypatch):
    from api.shared.persistence.pipeline_jobs import RetryJobOutcome, RetryJobResult

    service = FakePipelineService()
    service.retry_error = JobCancelledError("cancel won")
    _patch_retry_transition(
        monkeypatch,
        RetryJobResult(
            outcome=RetryJobOutcome.RETRIED,
            job={
                "job_id": "job-1",
                "filename": "a.pdf",
                "subject_id": "s1",
                "status": PipelineStatus.QUEUED.value,
                "retry_count": 2,
                "source_s3_key": "uploads/x.pdf",
            },
        ),
    )

    async def _fake_compensate(job_id, user_id, *, retry_count, retryable):
        assert retryable is False
        return RetryCompensationResult(
            outcome=RetryCompensationOutcome.CANCEL_REQUESTED,
            status=PipelineStatus.QUEUED.value,
            retry_count=retry_count,
            cancel_requested=True,
        )

    monkeypatch.setattr(pipeline, "compensate_failed_retry_reschedule", _fake_compensate)
    response = _build_client(service).post("/api/v1/pipeline/jobs/job-1/retry")
    assert response.status_code == 409
    assert response.json()["error"]["meta"]["state"] == "cancel_requested"
