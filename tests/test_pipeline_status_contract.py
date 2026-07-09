from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_content_pipeline_availability
from api.domains.content_pipeline import router as pipeline
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


def _build_client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.limiter = limiter
    app.include_router(pipeline.router)
    app.dependency_overrides[get_content_pipeline_availability] = lambda: {
        "available": True,
        "service_initialized": True,
        "error": None,
        "src": "test",
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr(pipeline, "is_admin_request", lambda *a, **k: True)


def test_pipeline_status_openapi_schema_is_not_untyped_dict():
    client = _build_client()

    schema = client.app.openapi()
    response_schema = schema["paths"]["/api/v1/pipeline/status"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert response_schema["$ref"].endswith("StandardResponse_PipelineRuntimeStatusResponse_")


def test_pipeline_status_returns_runtime_status():
    client = _build_client()

    response = client.get("/api/v1/pipeline/status")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "available": True,
        "service_initialized": True,
    }
