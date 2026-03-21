from api.core.exercise_gen import _is_retryable_llm_error


def test_is_retryable_llm_error_detects_html_gateway_failures():
    error = RuntimeError("<!DOCTYPE html><html><body>Hugging Face - Sorry, there is an error on our side.</body></html>")

    assert _is_retryable_llm_error(error) is True


def test_is_retryable_llm_error_ignores_non_transient_validation_errors():
    error = ValueError("LLM returned invalid type: <class 'str'>")

    assert _is_retryable_llm_error(error) is False
