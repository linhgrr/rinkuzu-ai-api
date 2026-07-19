def is_expected_degraded_readiness(status_code: int, payload: object) -> bool:
    if status_code != 503 or not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict) or error.get("code") != "service_unavailable":
        return False
    meta = error.get("meta")
    return isinstance(meta, dict) and meta.get("ready") is False
