import asyncio
import time

import pytest

from api.core.content_pipeline.application.stages.execution import run_process_stage
from api.core.content_pipeline.domain.errors import PipelineStageTimeoutError


def _sleep_and_return(delay: float, value: str) -> str:
    time.sleep(delay)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_run_process_stage_returns_result():
    result = asyncio.run(
        run_process_stage(
            "tests.core.content_pipeline.test_execution_stage:_sleep_and_return",
            0.01,
            "ok",
            stage_name="process_success",
            timeout_sec=1.0,
        )
    )

    assert result == "ok"


def test_run_process_stage_times_out():
    with pytest.raises(PipelineStageTimeoutError):
        asyncio.run(
            run_process_stage(
                "tests.core.content_pipeline.test_execution_stage:_sleep_and_return",
                1.0,
                "slow",
                stage_name="process_timeout",
                timeout_sec=0.05,
            )
        )


def test_run_process_stage_surfaces_process_errors():
    with pytest.raises(RuntimeError, match="process_failure failed in isolated process"):
        asyncio.run(
            run_process_stage(
                "tests.core.content_pipeline.test_execution_stage:_raise_runtime_error",
                stage_name="process_failure",
                timeout_sec=1.0,
            )
        )
