from types import SimpleNamespace

import pytest

from api.config import Settings
from api.core.content_pipeline.infrastructure.llm.openai_responses import (
    ProviderConfigError,
    build_provider_config,
    normalize_openai_base_url,
)


def test_normalize_openai_base_url_appends_v1():
    assert normalize_openai_base_url(None) == "https://api.openai.com/v1"
    assert normalize_openai_base_url("http://localhost:6969") == "http://localhost:6969/v1"
    assert normalize_openai_base_url("http://localhost:6969/v1") == "http://localhost:6969/v1"


def test_build_provider_config_requires_openai_settings(monkeypatch):
    monkeypatch.setattr(
        "api.core.content_pipeline.infrastructure.llm.openai_responses.get_settings",
        lambda: SimpleNamespace(
            openai_base_url="",
            openai_api_key="",
            openai_model="",
            content_pipeline_llm_request_timeout_sec=180,
            llm_max_retries=2,
        ),
    )

    with pytest.raises(ProviderConfigError):
        build_provider_config()


def test_build_provider_config_disables_sdk_retries_for_pipeline(monkeypatch):
    monkeypatch.setattr(
        "api.core.content_pipeline.infrastructure.llm.openai_responses.get_settings",
        lambda: SimpleNamespace(
            openai_base_url="https://api.openai.com",
            openai_api_key="test-key",
            openai_model="gpt-4.1-mini",
            content_pipeline_llm_request_timeout_sec=180,
        ),
    )

    config = build_provider_config()

    assert config.request_timeout_sec == 180
    assert config.max_retries == 0


def test_object_storage_client_endpoint_prefers_external_in_dev():
    settings = Settings.model_construct(
        environment="dev",
        object_storage_endpoint_internal="object-storage.objectstorage-system.svc.cluster.local",
        object_storage_endpoint_external="objectstorageapi.ap-southeast-1.clawcloudrun.com",
    )

    assert (
        settings.object_storage_client_endpoint
        == "https://objectstorageapi.ap-southeast-1.clawcloudrun.com"
    )


def test_object_storage_client_endpoint_prefers_internal_outside_dev():
    settings = Settings.model_construct(
        environment="prod",
        object_storage_endpoint_internal="object-storage.objectstorage-system.svc.cluster.local",
        object_storage_endpoint_external="objectstorageapi.ap-southeast-1.clawcloudrun.com",
    )

    assert (
        settings.object_storage_client_endpoint
        == "http://object-storage.objectstorage-system.svc.cluster.local"
    )
