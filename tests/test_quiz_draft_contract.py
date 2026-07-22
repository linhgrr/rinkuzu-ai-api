from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.dependencies import get_current_user
from api.domains.quiz import router as quiz_router
from api.exceptions import register_exception_handlers
from api.rate_limit import limiter


def _build_client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.limiter = limiter
    app.include_router(quiz_router.drafts_router)
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr(quiz_router, "is_admin_request", lambda *a, **k: True)


def test_quiz_draft_openapi_schemas_are_not_untyped_dict():
    client = _build_client()

    schema = client.app.openapi()
    paths = schema["paths"]

    assert paths["/api/v1/quiz/drafts"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("StandardResponse_QuizDraftListResponse_")
    assert paths["/api/v1/quiz/drafts"]["post"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("StandardResponse_QuizDraftSingleResponse_")
    assert paths["/api/v1/quiz/drafts/manual"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("StandardResponse_QuizDraftSingleResponse_")

    draft_path = paths["/api/v1/quiz/drafts/{draft_id}"]
    for method in ("get", "patch", "delete"):
        assert draft_path[method]["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ].endswith("StandardResponse_QuizDraftSingleResponse_")

    assert paths["/api/v1/quiz/drafts/{draft_id}/submit"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("StandardResponse_QuizDraftSingleResponse_")

    response_schema = schema["components"]["schemas"]["QuizDraftResponseData"]
    assert set(response_schema["required"]) == {
        "category_id",
        "created_at",
        "description",
        "draft_id",
        "error",
        "expires_at",
        "pdf",
        "prompt",
        "progress",
        "question_count",
        "questions",
        "revision",
        "is_private",
        "source_type",
        "status",
        "submitted_quiz_id",
        "title",
        "updated_at",
    }
    assert set(response_schema["properties"]["status"]["enum"]) == {
        "drafting",
        "queued",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "submitted",
        "expired",
    }
    assert set(schema["components"]["schemas"]["QuizDraftPdfInfo"]["required"]) == {
        "file_name",
        "file_size",
        "page_count",
        "s3_key",
    }
    assert schema["components"]["schemas"]["QuizDraftListResponse"]["required"] == ["drafts"]
