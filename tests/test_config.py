from pydantic import ValidationError
import pytest

from api.config import Settings

# >=32, not a placeholder, not all-same, not a pure short cycle.
_VALID_TOKEN = "test-internal-service-token-xx01"


def test_pipeline_resilience_settings_have_safe_defaults():
    # Explicit None overrides any ambient env so defaults stay testable in dev.
    s = Settings(environment="dev", internal_service_token=None)
    assert s.content_pipeline_reaper_interval_sec == 60
    assert s.content_pipeline_job_stalled_after_sec == 900
    assert s.content_pipeline_recovery_max_age_sec == 3600
    assert s.content_pipeline_dedup_window_sec == 30
    assert s.content_pipeline_max_retry_count == 3
    # stalled threshold MUST be stricter than the "delayed" UX threshold
    assert s.content_pipeline_job_stalled_after_sec > s.content_pipeline_job_delayed_after_sec


def test_internal_service_token_missing_allowed_in_non_prod():
    s = Settings(environment="dev", internal_service_token=None)
    assert s.internal_service_token is None

    s_staging = Settings(environment="staging", internal_service_token=None)
    assert s_staging.internal_service_token is None


def test_internal_service_token_blank_becomes_none_in_non_prod():
    s = Settings(environment="dev", internal_service_token="   ")
    assert s.internal_service_token is None


def test_internal_service_token_trims_and_accepts_valid():
    padded = f"  {_VALID_TOKEN}  "
    s = Settings(environment="dev", internal_service_token=padded)
    assert s.internal_service_token == _VALID_TOKEN


def test_internal_service_token_rejects_short_without_echoing_value():
    short = "too-short-to-be-valid"
    with pytest.raises(ValidationError) as exc_info:
        Settings(environment="dev", internal_service_token=short)
    message = str(exc_info.value)
    assert short not in message
    assert "at least 32" in message


@pytest.mark.parametrize(
    "placeholder",
    [
        "replace-me",
        "change-me",
        "your-token",
        "example",
        "default",
        "Replace-Me",
        # Long (>=32) packed markers must also fail as placeholders.
        "replace-me" * 4,
        "CHANGE-ME" * 4,
        "your-token" * 4,
        "example" * 5,
        "default" * 5,
        # Obvious all-same / short-cycle repeated values.
        "a" * 32,
        "AbAb" * 8,
    ],
)
def test_internal_service_token_rejects_placeholders_without_echoing_value(placeholder: str):
    with pytest.raises(ValidationError) as exc_info:
        Settings(environment="dev", internal_service_token=placeholder)
    message = str(exc_info.value)
    assert placeholder not in message
    assert "placeholder" in message.lower()


def test_internal_service_token_accepts_normal_random_token():
    # Looks random enough: not a known marker, not all-same, not pure short cycle.
    token = "n0rmal-rand0m-service-token-ok01"  # pragma: allowlist secret
    s = Settings(environment="dev", internal_service_token=token)
    assert s.internal_service_token == token


def test_internal_service_token_required_in_prod():
    with pytest.raises(ValidationError) as exc_info:
        Settings(environment="prod", internal_service_token=None)
    message = str(exc_info.value)
    assert "required in production" in message


def test_internal_service_token_prod_rejects_blank():
    with pytest.raises(ValidationError) as exc_info:
        Settings(environment="prod", internal_service_token="  ")
    message = str(exc_info.value)
    assert "required in production" in message


def test_internal_service_token_prod_accepts_valid():
    s = Settings(environment="prod", internal_service_token=_VALID_TOKEN)
    assert s.internal_service_token == _VALID_TOKEN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DEV", "dev"),
        (" development ", "dev"),
        ("Staging", "staging"),
        ("PROD", "prod"),
        (" production ", "prod"),
    ],
)
def test_environment_aliases_are_normalized(raw: str, expected: str):
    s = Settings(environment=raw, internal_service_token=_VALID_TOKEN)
    assert s.environment == expected


@pytest.mark.parametrize("raw", ["", "test", "local", 123])
def test_environment_rejects_unknown_values(raw: object):
    with pytest.raises(ValidationError, match="environment must be"):
        Settings(environment=raw, internal_service_token=_VALID_TOKEN)


def test_production_alias_requires_internal_service_token():
    with pytest.raises(ValidationError, match="required in production"):
        Settings(environment="production", internal_service_token=None)


def test_internal_service_token_non_string_is_validation_error():
    with pytest.raises(ValidationError, match="must be a string"):
        Settings(environment="dev", internal_service_token=123)
