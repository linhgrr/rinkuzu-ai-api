from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.exceptions import register_exception_handlers
from api.routers import admin_ocr_keys


def _test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(admin_ocr_keys.router)
    return app


def test_admin_ocr_keys_list_returns_masked_keys(monkeypatch):
    monkeypatch.setattr(admin_ocr_keys.mongo_store, "is_available", lambda: True)
    monkeypatch.setattr(
        "api.dependencies.get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token="secret"),
    )
    monkeypatch.setattr(
        admin_ocr_keys,
        "get_settings",
        lambda: SimpleNamespace(ocr_api_key="env-fallback-key"),  # pragma: allowlist secret
    )

    async def fake_list_ocr_provider_keys():
        return [
            {
                "key_id": "key-1",
                "provider": "landingai",
                "label": "Primary",
                "masked_key": "land••••••••1234",
                "enabled": True,
                "health_status": "healthy",
                "priority": 10,
                "success_count": 2,
                "failure_count": 1,
                "last_used_at": None,
                "last_success_at": None,
                "last_error_at": None,
                "last_error_code": None,
                "last_error_message": None,
                "created_at": "2026-07-10T00:00:00Z",
                "updated_at": "2026-07-10T00:00:00Z",
                "encrypted_key": "must-not-leak",
                "api_key": "must-not-leak",  # pragma: allowlist secret
            }
        ]

    monkeypatch.setattr(admin_ocr_keys, "list_ocr_provider_keys", fake_list_ocr_provider_keys)

    response = TestClient(_test_app()).get(
        "/api/v1/admin/ocr-keys",
        headers={
            "x-user-id": "admin-1",
            "x-user-role": "admin",
            "x-service-token": "secret",
        },
    )

    assert response.status_code == 200
    body = response.json()
    key = body["data"]["keys"][0]
    assert key["masked_key"] == "land••••••••1234"
    assert "encrypted_key" not in key
    assert "api_key" not in key
    assert body["data"]["fallback_env_configured"] is True


def test_admin_ocr_key_test_returns_masked_key(monkeypatch):
    success_calls: list[str] = []
    monkeypatch.setattr(admin_ocr_keys.mongo_store, "is_available", lambda: True)
    monkeypatch.setattr(
        "api.dependencies.get_settings",
        lambda: SimpleNamespace(environment="prod", internal_service_token="secret"),
    )
    monkeypatch.setattr(
        admin_ocr_keys,
        "decrypt_ocr_key",
        lambda _encrypted_key: "raw-test-key",  # pragma: allowlist secret
    )

    async def fake_load_ocr_provider_key_secret(*, key_id: str):
        assert key_id == "key-1"
        return {
            "key_id": "key-1",
            "encrypted_key": "encrypted-secret",
            "masked_key": "land••••••••1234",
        }

    async def fake_check_ocr_api_key(api_key: str):
        assert api_key == "raw-test-key"  # pragma: allowlist secret
        return object()

    async def fake_record_ocr_key_success(*, key_id: str):
        success_calls.append(key_id)

    async def fake_load_ocr_provider_key(*, key_id: str):
        assert key_id == "key-1"
        return {
            "key_id": "key-1",
            "provider": "landingai",
            "label": "Primary",
            "masked_key": "land••••••••1234",
            "enabled": True,
            "health_status": "healthy",
            "priority": 10,
            "success_count": 3,
            "failure_count": 1,
            "last_used_at": None,
            "last_success_at": None,
            "last_error_at": None,
            "last_error_code": None,
            "last_error_message": None,
            "created_at": "2026-07-10T00:00:00Z",
            "updated_at": "2026-07-10T00:00:00Z",
            "encrypted_key": "must-not-leak",
            "api_key": "must-not-leak",  # pragma: allowlist secret
        }

    monkeypatch.setattr(
        admin_ocr_keys,
        "load_ocr_provider_key_secret",
        fake_load_ocr_provider_key_secret,
    )
    monkeypatch.setattr(admin_ocr_keys, "check_ocr_api_key", fake_check_ocr_api_key)
    monkeypatch.setattr(admin_ocr_keys, "record_ocr_key_success", fake_record_ocr_key_success)
    monkeypatch.setattr(admin_ocr_keys, "load_ocr_provider_key", fake_load_ocr_provider_key)

    response = TestClient(_test_app()).post(
        "/api/v1/admin/ocr-keys/key-1/test",
        headers={
            "x-user-id": "admin-1",
            "x-user-role": "admin",
            "x-service-token": "secret",
        },
    )

    assert response.status_code == 200
    body = response.json()
    key = body["data"]["key"]
    assert body["data"]["ok"] is True
    assert key["masked_key"] == "land••••••••1234"
    assert "encrypted_key" not in key
    assert "api_key" not in key
    assert success_calls == ["key-1"]
