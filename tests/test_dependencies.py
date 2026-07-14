from types import SimpleNamespace

import pytest

from api import dependencies
from api.exceptions import AppError, ServiceUnavailableError


def test_get_content_pipeline_service_reads_app_state():
    service = object()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(content_pipeline_service=service))
    )

    resolved = dependencies.get_content_pipeline_service(request)

    assert resolved is service


def test_get_content_pipeline_service_raises_when_missing():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    with pytest.raises(ServiceUnavailableError) as exc_info:
        dependencies.get_content_pipeline_service(request)

    assert "ContentPipelineService" in str(exc_info.value)


def test_get_content_pipeline_availability_reads_app_state():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                content_processor_available=True,
                content_processor_error="boom",
                content_processor_src="fixtures/content-pipeline-runtime",
            )
        )
    )

    availability = dependencies.get_content_pipeline_availability(request)

    assert availability == {
        "available": True,
        "error": "boom",
        "src": "fixtures/content-pipeline-runtime",
        "service_initialized": False,
    }


def test_get_current_user_requires_service_token_in_non_dev(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token=None),
    )

    with pytest.raises(AppError) as exc_info:
        dependencies.get_current_user(x_user_id="user-1", x_service_token=None)

    assert exc_info.value.status_code == 500


def test_get_current_user_accepts_valid_service_token(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token="secret"),
    )

    assert dependencies.get_current_user(x_user_id="user-1", x_service_token="secret") == "user-1"


def test_get_current_user_requires_token_even_in_dev(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="dev", internal_service_token="secret"),
    )

    with pytest.raises(AppError) as exc_info:
        dependencies.get_current_user(x_user_id="user-1", x_service_token=None)

    assert exc_info.value.status_code == 401


def test_get_current_user_rejects_mismatched_token(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="dev", internal_service_token="secret"),
    )

    with pytest.raises(AppError) as exc_info:
        dependencies.get_current_user(x_user_id="user-1", x_service_token="wrong")

    assert exc_info.value.status_code == 401


def test_get_current_user_requires_user_id_header(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token="secret"),
    )

    with pytest.raises(AppError) as exc_info:
        dependencies.get_current_user(x_user_id=None, x_service_token="secret")

    assert exc_info.value.status_code == 401


def test_get_current_admin_user_accepts_admin_role(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token="secret"),
    )

    assert (
        dependencies.get_current_admin_user(
            x_user_id="admin-1",
            x_user_role="admin",
            x_service_token="secret",
        )
        == "admin-1"
    )


def test_get_current_admin_user_rejects_non_admin_role(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token="secret"),
    )

    with pytest.raises(AppError) as exc_info:
        dependencies.get_current_admin_user(
            x_user_id="user-1",
            x_user_role="student",
            x_service_token="secret",
        )

    assert exc_info.value.status_code == 403
