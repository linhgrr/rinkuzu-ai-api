from types import SimpleNamespace

import pytest

from api import dependencies


def test_get_content_pipeline_service_reads_app_state():
    service = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(content_pipeline_service=service)))

    resolved = dependencies.get_content_pipeline_service(request)

    assert resolved is service


def test_get_content_pipeline_service_raises_when_missing():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    with pytest.raises(Exception) as exc_info:
        dependencies.get_content_pipeline_service(request)

    assert "ContentPipelineService" in str(exc_info.value)


def test_get_content_pipeline_availability_reads_app_state():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                content_processor_available=True,
                content_processor_error="boom",
                content_processor_src="/tmp/content-processor/src",
            )
        )
    )

    availability = dependencies.get_content_pipeline_availability(request)

    assert availability == {
        "available": True,
        "error": "boom",
        "src": "/tmp/content-processor/src",
    }
