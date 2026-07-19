from tests.perf_readiness import is_expected_degraded_readiness


def test_expected_degraded_readiness_requires_exact_contract() -> None:
    assert is_expected_degraded_readiness(
        503,
        {
            "success": False,
            "error": {
                "code": "service_unavailable",
                "message": "Service unavailable",
                "meta": {"ready": False, "mongo_available": False},
            },
        },
    )


def test_expected_degraded_readiness_rejects_wrong_status_or_payload() -> None:
    assert not is_expected_degraded_readiness(200, {"data": {"ready": True}})
    assert not is_expected_degraded_readiness(
        503,
        {"error": {"code": "internal_error", "meta": {"ready": False}}},
    )
    assert not is_expected_degraded_readiness(
        503,
        {"error": {"code": "service_unavailable", "meta": {"ready": True}}},
    )
