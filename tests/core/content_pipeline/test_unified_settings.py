from types import SimpleNamespace

import pytest

from api.config import Settings
from api.core.shared import llm as llm_module
from api.domains.content_pipeline.infrastructure.llm.structured_generation import (
    ProviderConfigError,
    build_provider_config,
)


def test_normalize_llm_base_url_requires_explicit_value():
    with pytest.raises(llm_module.LLMConfigurationError, match="LLM_BASE_URL"):
        llm_module.normalize_llm_base_url(None)

    assert llm_module.normalize_llm_base_url("http://localhost:6969") == "http://localhost:6969"
    assert llm_module.normalize_llm_base_url("http://localhost:6969/") == "http://localhost:6969"


def test_build_llm_provider_config_infers_deepseek_provider(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="https://api.deepseek.com",
            llm_api_key="test-key",  # pragma: allowlist secret
            llm_model="deepseek-v4-flash",
            llm_custom_provider=None,
            llm_timeout_sec=120,
        ),
    )

    config = llm_module.build_llm_provider_config()

    assert config.model == "deepseek-v4-flash"
    assert config.custom_llm_provider == "deepseek"


def test_build_llm_provider_config_allows_explicit_custom_provider(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="https://senator-gigolo-stark.ngrok-free.dev/v1",
            llm_api_key="test-key",  # pragma: allowlist secret
            llm_model="vip",
            llm_custom_provider="openai",
            llm_timeout_sec=120,
        ),
    )

    config = llm_module.build_llm_provider_config()

    assert config.model == "vip"
    assert config.custom_llm_provider == "openai"


def test_build_provider_config_requires_llm_settings(monkeypatch):
    monkeypatch.setattr(
        "api.domains.content_pipeline.infrastructure.llm.structured_generation.get_settings",
        lambda: SimpleNamespace(
            llm_base_url="",
            llm_api_key="",
            llm_model="",
            content_pipeline_llm_request_timeout_sec=180,
        ),
    )
    monkeypatch.setattr(
        "api.core.shared.llm.get_settings",
        lambda: SimpleNamespace(
            llm_base_url="",
            llm_api_key="",
            llm_model="",
            llm_timeout_sec=120,
        ),
    )

    with pytest.raises(ProviderConfigError):
        build_provider_config()


def test_build_provider_config_uses_pipeline_timeout(monkeypatch):
    monkeypatch.setattr(
        "api.domains.content_pipeline.infrastructure.llm.structured_generation.get_settings",
        lambda: SimpleNamespace(
            llm_base_url="https://api.deepseek.com",
            llm_api_key="test-key",  # pragma: allowlist secret
            llm_model="deepseek-v4-flash",
            content_pipeline_llm_request_timeout_sec=180,
        ),
    )

    monkeypatch.setattr(
        "api.core.shared.llm.get_settings",
        lambda: SimpleNamespace(
            llm_base_url="https://api.deepseek.com",
            llm_api_key="test-key",  # pragma: allowlist secret
            llm_model="deepseek-v4-flash",
            llm_timeout_sec=120,
        ),
    )

    config = build_provider_config()

    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"
    assert config.timeout_sec == 180
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
