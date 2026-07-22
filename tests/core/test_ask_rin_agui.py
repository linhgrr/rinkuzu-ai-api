import asyncio
import json
from typing import Any

from ag_ui.core import RunAgentInput
import pytest

from api.domains.assistant import agui
from api.domains.assistant.context_tokens import ExerciseContext
from api.shared.llm import LLMOutputTruncatedError


def _input(*, token: str = "t" * 32, run_id: str = "run_123456") -> RunAgentInput:
    return RunAgentInput.model_validate(
        {
            "threadId": "quiz:question-hash",
            "runId": run_id,
            "state": None,
            "messages": [{"id": "user-message", "role": "user", "content": "Giải thích giúp mình"}],
            "tools": [],
            "context": [],
            "forwardedProps": {"exerciseContextToken": token},
        }
    )


def _context() -> ExerciseContext:
    return ExerciseContext(
        context_id="quiz:question-hash",
        user_id="user-1",
        question="2 + 2 bằng bao nhiêu?",
        options=["3", "4", "5", "6"],
    )


async def _events(response: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        events.extend(
            json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
        )
    return events


class _Service:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas

    async def create_delta_stream(self, _context: Any) -> Any:
        async def stream():
            for delta in self.deltas:
                yield delta

        return stream()


class _FailingService:
    async def create_delta_stream(self, _context: Any) -> Any:
        raise ValueError("unsafe message")


class _CancellingService:
    async def create_delta_stream(self, _context: Any) -> Any:
        async def stream():
            yield "Đang giải"
            raise asyncio.CancelledError

        return stream()


class _TruncatedService:
    async def create_delta_stream(self, _context: Any) -> Any:
        async def stream():
            yield "Đang giải đến đáp án D"
            raise LLMOutputTruncatedError("response length limit")

        return stream()


def test_agui_input_requires_signed_context_and_matching_thread() -> None:
    input_data = _input()

    assert agui.read_exercise_context_token(input_data) == "t" * 32
    assert agui.latest_user_message(input_data) == "Giải thích giúp mình"
    agui.validate_run_identity(input_data, _context())

    input_data.thread_id = "quiz:another-question"
    with pytest.raises(ValueError, match="threadId"):
        agui.validate_run_identity(input_data, _context())


def test_agui_accepts_assistant_ui_seven_character_run_id() -> None:
    agui.validate_run_identity(_input(run_id="Ab3xYz9"), _context())


@pytest.mark.asyncio
async def test_agui_stream_emits_typed_lifecycle_and_persists_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished: list[dict[str, Any]] = []

    async def begin_turn(**_kwargs: Any) -> tuple[dict[str, str], None]:
        return {"conversation_id": "conversation-1"}, None

    async def load_model_history(*_args: Any) -> list[dict[str, str]]:
        return []

    async def finish_turn(**kwargs: Any) -> str:
        finished.append(kwargs)
        return agui.assistant_message_id("user-1", "run_123456")

    monkeypatch.setattr(agui, "begin_turn", begin_turn)
    monkeypatch.setattr(agui, "load_model_history", load_model_history)
    monkeypatch.setattr(agui, "finish_turn", finish_turn)

    response = await agui.create_agui_response(
        request_accept="text/event-stream",
        input_data=_input(),
        user_id="user-1",
        context=_context(),
        service=_Service(["Đáp ", "án là 4."]),
    )
    events = await _events(response)

    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "STATE_SNAPSHOT",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert events[0] == {
        "type": "RUN_STARTED",
        "threadId": "quiz:question-hash",
        "runId": "run_123456",
    }
    assert events[-1]["result"]["conversationId"] == "conversation-1"
    assert finished == [
        {
            "user_id": "user-1",
            "client_request_id": "run_123456",
            "content": "Đáp án là 4.",
            "interrupted": False,
        }
    ]


@pytest.mark.asyncio
async def test_agui_replay_uses_same_message_id_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def begin_turn(**_kwargs: Any) -> tuple[dict[str, str], str]:
        return {"conversation_id": "conversation-1"}, "Câu trả lời cũ"

    monkeypatch.setattr(agui, "begin_turn", begin_turn)
    response = await agui.create_agui_response(
        request_accept=None,
        input_data=_input(),
        user_id="user-1",
        context=_context(),
        service=_Service([]),
    )
    events = await _events(response)

    starts = [event for event in events if event["type"] == "TEXT_MESSAGE_START"]
    assert starts[0]["messageId"] == agui.assistant_message_id("user-1", "run_123456")
    assert events[-1]["type"] == "RUN_FINISHED"
    assert events[-1]["result"]["replayed"] is True


@pytest.mark.asyncio
async def test_agui_preflight_failure_refunds_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refunded: list[dict[str, str]] = []

    async def begin_turn(**_kwargs: Any) -> tuple[dict[str, str], None]:
        return {"conversation_id": "conversation-1"}, None

    async def load_model_history(*_args: Any) -> list[dict[str, str]]:
        return []

    async def refund_turn(**kwargs: str) -> None:
        refunded.append(kwargs)

    monkeypatch.setattr(agui, "begin_turn", begin_turn)
    monkeypatch.setattr(agui, "load_model_history", load_model_history)
    monkeypatch.setattr(agui, "refund_turn", refund_turn)

    with pytest.raises(ValueError, match="unsafe message"):
        await agui.create_agui_response(
            request_accept=None,
            input_data=_input(),
            user_id="user-1",
            context=_context(),
            service=_FailingService(),
        )

    assert refunded == [{"user_id": "user-1", "client_request_id": "run_123456"}]


@pytest.mark.asyncio
async def test_agui_cancellation_persists_partial_answer_as_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished: list[dict[str, Any]] = []

    async def begin_turn(**_kwargs: Any) -> tuple[dict[str, str], None]:
        return {"conversation_id": "conversation-1"}, None

    async def load_model_history(*_args: Any) -> list[dict[str, str]]:
        return []

    async def finish_turn(**kwargs: Any) -> str:
        finished.append(kwargs)
        return "assistant-message"

    monkeypatch.setattr(agui, "begin_turn", begin_turn)
    monkeypatch.setattr(agui, "load_model_history", load_model_history)
    monkeypatch.setattr(agui, "finish_turn", finish_turn)

    response = await agui.create_agui_response(
        request_accept=None,
        input_data=_input(),
        user_id="user-1",
        context=_context(),
        service=_CancellingService(),
    )
    with pytest.raises(asyncio.CancelledError):
        await _events(response)

    assert finished == [
        {
            "user_id": "user-1",
            "client_request_id": "run_123456",
            "content": "Đang giải",
            "interrupted": True,
        }
    ]


@pytest.mark.asyncio
async def test_agui_output_truncation_marks_partial_answer_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished: list[dict[str, Any]] = []

    async def begin_turn(**_kwargs: Any) -> tuple[dict[str, str], None]:
        return {"conversation_id": "conversation-1"}, None

    async def load_model_history(*_args: Any) -> list[dict[str, str]]:
        return []

    async def finish_turn(**kwargs: Any) -> str:
        finished.append(kwargs)
        return "assistant-message"

    monkeypatch.setattr(agui, "begin_turn", begin_turn)
    monkeypatch.setattr(agui, "load_model_history", load_model_history)
    monkeypatch.setattr(agui, "finish_turn", finish_turn)

    response = await agui.create_agui_response(
        request_accept=None,
        input_data=_input(),
        user_id="user-1",
        context=_context(),
        service=_TruncatedService(),
    )
    events = await _events(response)

    assert [event["type"] for event in events[-2:]] == ["TEXT_MESSAGE_END", "RUN_ERROR"]
    assert events[-1]["code"] == "ask_rin_run_failed"
    assert finished == [
        {
            "user_id": "user-1",
            "client_request_id": "run_123456",
            "content": "Đang giải đến đáp án D",
            "interrupted": True,
        }
    ]
