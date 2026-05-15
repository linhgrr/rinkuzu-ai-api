from datetime import UTC, datetime

from bson import ObjectId
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from api import main as _api_main  # noqa: F401  # register global JSON encoders


def test_orjson_response_handles_datetime_and_object_id():
    app = FastAPI(default_response_class=ORJSONResponse)

    @app.get("/probe")
    async def probe():
        return {
            "timestamp": datetime(2025, 1, 1, 12, 30, tzinfo=UTC),
            "object_id": ObjectId("507f1f77bcf86cd799439011"),
        }

    client = TestClient(app)
    response = client.get("/probe")

    assert response.status_code == 200
    payload = response.json()
    assert payload["timestamp"].startswith("2025-01-01T12:30:00")
    assert payload["object_id"] == "507f1f77bcf86cd799439011"
