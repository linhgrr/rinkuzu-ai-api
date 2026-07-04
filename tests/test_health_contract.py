from api.main import app


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
