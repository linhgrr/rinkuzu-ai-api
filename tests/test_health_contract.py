from api.main import app

_PROXY_HEADER_NAMES = frozenset({"x-service-token", "x-user-id", "x-user-role"})
_PUBLIC_OPERATIONS = {
    ("get", "/api/ready"),
    ("get", "/api/health"),
    ("get", "/api/info"),
    ("get", "/api/v1/pipeline/status"),
}


def _iter_operations(schema):
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method.startswith("x-") or not isinstance(operation, dict):
                continue
            yield method, path, operation


def test_ready_health_and_info_openapi_schemas_are_typed():
    schema = app.openapi()

    ready_schema = schema["paths"]["/api/ready"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    health_schema = schema["paths"]["/api/health"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    info_schema = schema["paths"]["/api/info"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert ready_schema["$ref"].endswith("StandardResponse_ReadinessResponse_")
    assert health_schema["$ref"].endswith("StandardResponse_ReadinessResponse_")
    assert info_schema["$ref"].endswith("StandardResponse_InfoResponse_")


def test_openapi_advertises_service_token_scheme_and_requiredness():
    schema = app.openapi()

    schemes = schema["components"]["securitySchemes"]
    assert "XServiceTokenHeader" in schemes
    assert schemes["XServiceTokenHeader"]["name"] == "x-service-token"
    assert schemes["XServiceTokenHeader"]["in"] == "header"
    assert schemes["XServiceTokenHeader"]["type"] == "apiKey"

    # Global security requires both proxy headers for protected operations.
    assert {"XUserIdHeader": [], "XServiceTokenHeader": []} in schema["security"]

    for _method, path in _PUBLIC_OPERATIONS:
        operation = schema["paths"][path]["get"]
        assert operation.get("security") == [], path

    # A typical protected operation inherits global security (no empty override).
    protected = schema["paths"]["/api/v1/pipeline/jobs"]["get"]
    assert "security" not in protected or protected["security"] != []


def test_openapi_protected_proxy_headers_are_required():
    """Every explicit proxy header on a protected op must be required=true.

    Public ops stay security=[] and must not be falsely marked required for
    these headers (they typically omit them entirely).
    """
    schema = app.openapi()
    optional_protected: list[str] = []
    public_with_required_proxy: list[str] = []

    for method, path, operation in _iter_operations(schema):
        operation_key = (method, path)
        is_public = operation_key in _PUBLIC_OPERATIONS
        if operation.get("security") == []:
            assert is_public, f"unexpected public operation: {method.upper()} {path}"
        for param in operation.get("parameters") or []:
            if not isinstance(param, dict):
                continue
            name = (param.get("name") or "").lower()
            if param.get("in") != "header" or name not in _PROXY_HEADER_NAMES:
                continue
            if is_public and param.get("required") is True:
                public_with_required_proxy.append(f"{method.upper()} {path} {name}")
            elif not is_public and param.get("required") is not True:
                optional_protected.append(f"{method.upper()} {path} {name}")
            elif not is_public and param.get("schema", {}).get("type") != "string":
                optional_protected.append(
                    f"{method.upper()} {path} {name} has nullable/non-string schema"
                )

        if is_public:
            assert operation.get("security") == [], f"{method.upper()} {path}"

    assert optional_protected == [], (
        "protected explicit proxy headers must be required=true: " + ", ".join(optional_protected)
    )
    assert public_with_required_proxy == [], (
        "public ops must not force proxy headers required: " + ", ".join(public_with_required_proxy)
    )
