from types import SimpleNamespace

from api import rate_limit


def _make_request(headers: dict[str, str], client_host: str = "1.2.3.4"):
    # slowapi.util.get_remote_address reads request.client.host.
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=client_host))


def test_rate_limit_key_uses_user_id_when_token_valid(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request(
        {"x-user-id": "u-1", "x-service-token": "secret"},
    )

    assert rate_limit.rate_limit_key(request) == "user:u-1"


def test_rate_limit_key_falls_back_to_ip_when_token_missing(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request({"x-user-id": "u-1"})

    assert rate_limit.rate_limit_key(request) == "1.2.3.4"


def test_rate_limit_key_falls_back_to_ip_when_token_mismatch(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request(
        {"x-user-id": "u-1", "x-service-token": "wrong"},
    )

    assert rate_limit.rate_limit_key(request) == "1.2.3.4"


def test_rate_limit_key_falls_back_to_ip_when_token_unconfigured(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token=None),
    )

    request = _make_request(
        {"x-user-id": "u-1", "x-service-token": "anything"},
    )

    assert rate_limit.rate_limit_key(request) == "1.2.3.4"


def test_is_admin_request_requires_token_match(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request({"x-service-token": "secret"})
    token = rate_limit.set_current_rate_limit_request(request)
    try:
        assert rate_limit.is_admin_request() is True
    finally:
        rate_limit.reset_current_rate_limit_request(token)


def test_is_admin_request_returns_false_without_token(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request({})
    token = rate_limit.set_current_rate_limit_request(request)
    try:
        assert rate_limit.is_admin_request() is False
    finally:
        rate_limit.reset_current_rate_limit_request(token)


def test_is_admin_request_returns_false_on_mismatch(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request({"x-service-token": "wrong"})
    token = rate_limit.set_current_rate_limit_request(request)
    try:
        assert rate_limit.is_admin_request() is False
    finally:
        rate_limit.reset_current_rate_limit_request(token)


def test_is_admin_request_returns_false_when_token_unconfigured(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token=None),
    )

    request = _make_request({"x-service-token": "anything"})
    token = rate_limit.set_current_rate_limit_request(request)
    try:
        assert rate_limit.is_admin_request() is False
    finally:
        rate_limit.reset_current_rate_limit_request(token)


def test_rate_limit_key_rejects_length_mismatched_token(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "get_settings",
        lambda: SimpleNamespace(internal_service_token="secret"),
    )

    request = _make_request(
        {"x-user-id": "u-1", "x-service-token": "secret-extra"},
    )

    assert rate_limit.rate_limit_key(request) == "1.2.3.4"
