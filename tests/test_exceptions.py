from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.exceptions import AppError, register_exception_handlers


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/http")
    async def raise_http():
        raise HTTPException(status_code=418, detail="teapot")

    @app.get("/app")
    async def raise_app():
        raise AppError(
            code="conflict",
            message="Conflict",
            detail="app boom",
            status_code=409,
        )

    @app.get("/unexpected")
    async def raise_unexpected():
        raise RuntimeError("sensitive internals")

    @app.get("/items/{item_id}")
    async def read_item(item_id: int):
        return {"item_id": item_id}

    return app


def test_http_exception_is_normalized():
    client = TestClient(_build_app())

    response = client.get("/http")

    assert response.status_code == 418
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "upstream_error"
    assert payload["error"]["detail"] == "teapot"
    assert payload["error"]["message"] == "Request failed"


def test_app_error_is_normalized():
    client = TestClient(_build_app())

    response = client.get("/app")

    assert response.status_code == 409
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "conflict"
    assert payload["error"]["detail"] == "app boom"
    assert payload["error"]["message"] == "Conflict"


def test_validation_error_is_normalized():
    client = TestClient(_build_app())

    response = client.get("/items/not-an-int")

    assert response.status_code == 422
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Invalid request"
    assert payload["error"]["detail"] == "Request validation failed"
    assert isinstance(payload["error"]["meta"], list)
    assert payload["error"]["meta"]


def test_unexpected_error_is_sanitized():
    client = TestClient(_build_app(), raise_server_exceptions=False)

    response = client.get("/unexpected")

    assert response.status_code == 500
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "internal_error"
    assert "detail" not in payload["error"]
    assert payload["error"]["message"] == "Internal server error"
