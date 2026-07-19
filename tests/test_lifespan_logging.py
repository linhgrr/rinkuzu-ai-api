from types import SimpleNamespace

from loguru import logger

from api import lifespan


def test_log_llm_config_never_logs_api_key_material() -> None:
    secret = "super-secret-llm-api-key"  # pragma: allowlist secret
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")

    try:
        lifespan._log_llm_config(
            SimpleNamespace(
                llm_api_key=secret,
                llm_base_url="https://llm.example.com",
                llm_model="model",
                llm_custom_provider="provider",
                active_exercise_llm_model="exercise-model",
                llm_max_retries=2,
            )
        )
    finally:
        logger.remove(sink_id)

    output = "".join(messages)
    assert "api_key     = configured" in output
    assert secret not in output
    assert secret[:6] not in output
