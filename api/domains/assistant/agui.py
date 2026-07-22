"""AG-UI transport adapter for the Ask Rin-chan domain service."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi.responses import StreamingResponse

from api.shared.llm import SSE_STREAM_HEADERS
from api.shared.llm_usage import LlmAction

from .repository import (
    assistant_message_id,
    begin_turn,
    finish_turn,
    load_model_history,
    refund_turn,
)
from .service import AskRinChanService, AskRinRequestContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .context_tokens import ExerciseContext

_MAX_RUN_ID_LENGTH = 128
_MAX_INPUT_MESSAGES = 100
_MIN_CONTEXT_TOKEN_LENGTH = 32
_MAX_CONTEXT_TOKEN_LENGTH = 32_768
_MAX_USER_MESSAGE_LENGTH = 1_000


def read_exercise_context_token(input_data: RunAgentInput) -> str:
    props = input_data.forwarded_props
    if not isinstance(props, dict):
        raise TypeError("forwardedProps must be an object")
    token = props.get("exerciseContextToken")
    if not isinstance(token, str) or not (
        _MIN_CONTEXT_TOKEN_LENGTH <= len(token) <= _MAX_CONTEXT_TOKEN_LENGTH
    ):
        raise ValueError("forwardedProps.exerciseContextToken is required")
    return token


def validate_run_identity(input_data: RunAgentInput, context: ExerciseContext) -> None:
    if input_data.thread_id != context.context_id:
        raise ValueError("threadId does not match the signed exercise context")
    if not 1 <= len(input_data.run_id) <= _MAX_RUN_ID_LENGTH:
        raise ValueError("runId must contain 1-128 characters")
    if len(input_data.messages) > _MAX_INPUT_MESSAGES:
        raise ValueError(f"At most {_MAX_INPUT_MESSAGES} messages are accepted")


def latest_user_message(input_data: RunAgentInput) -> str:
    message = next(
        (item for item in reversed(input_data.messages) if item.role == "user"),
        None,
    )
    if message is None:
        raise ValueError("A user message is required")

    content = message.content
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(
            str(getattr(part, "text", ""))
            for part in content
            if getattr(part, "type", None) == "text"
        )
    text = text.strip()
    if not text or len(text) > _MAX_USER_MESSAGE_LENGTH:
        raise ValueError("The latest user message must contain 1-1000 text characters")
    return text


def _run_started_events(
    *,
    encoder: EventEncoder,
    input_data: RunAgentInput,
    conversation_id: str,
    context_id: str,
    response_message_id: str,
) -> tuple[str, ...]:
    return (
        encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
            )
        ),
        encoder.encode(
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot={
                    "conversationId": conversation_id,
                    "exerciseContextId": context_id,
                },
            )
        ),
        encoder.encode(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=response_message_id,
                role="assistant",
            )
        ),
    )


def _replay_events(
    *,
    encoder: EventEncoder,
    input_data: RunAgentInput,
    conversation_id: str,
    response_message_id: str,
    replay: str,
) -> tuple[str, ...]:
    events: list[str] = []
    if replay:
        events.append(
            encoder.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=response_message_id,
                    delta=replay,
                )
            )
        )
    events.extend(
        (
            encoder.encode(
                TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=response_message_id,
                )
            ),
            encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    result={"conversationId": conversation_id, "replayed": True},
                )
            ),
        )
    )
    return tuple(events)


async def _settle_interrupted_run(*, user_id: str, run_id: str, full_response: str) -> None:
    content = full_response.strip()
    if content:
        await finish_turn(
            user_id=user_id,
            client_request_id=run_id,
            content=content,
            interrupted=True,
        )
        return
    await refund_turn(user_id=user_id, client_request_id=run_id)


async def _fresh_run_events(
    *,
    encoder: EventEncoder,
    input_data: RunAgentInput,
    user_id: str,
    conversation_id: str,
    response_message_id: str,
    upstream: AsyncIterator[str],
) -> AsyncIterator[str]:
    full_response = ""
    try:
        async for delta in upstream:
            if not delta:
                continue
            full_response += delta
            yield encoder.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=response_message_id,
                    delta=delta,
                )
            )
        if not full_response.strip():
            raise RuntimeError("LLM returned an empty completion")
    except asyncio.CancelledError:
        await asyncio.shield(
            _settle_interrupted_run(
                user_id=user_id,
                run_id=input_data.run_id,
                full_response=full_response,
            )
        )
        raise
    except Exception as exc:
        await _settle_interrupted_run(
            user_id=user_id,
            run_id=input_data.run_id,
            full_response=full_response,
        )
        yield encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=response_message_id,
            )
        )
        yield encoder.encode(
            RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(exc) or "Rin-chan could not answer this message",
                code="ask_rin_run_failed",
            )
        )
        return

    persisted_message_id = await finish_turn(
        user_id=user_id,
        client_request_id=input_data.run_id,
        content=full_response.strip(),
        interrupted=False,
    )
    yield encoder.encode(
        TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=response_message_id,
        )
    )
    yield encoder.encode(
        RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
            result={
                "conversationId": conversation_id,
                "assistantMessageId": persisted_message_id,
            },
        )
    )


async def create_agui_response(
    *,
    request_accept: str | None,
    input_data: RunAgentInput,
    user_id: str,
    context: ExerciseContext,
    service: AskRinChanService,
) -> StreamingResponse:
    """Create a cancellable AG-UI run over the shared Ask Rin-chan service."""
    encoder = EventEncoder(accept=request_accept or "text/event-stream")
    message = latest_user_message(input_data)
    conversation, replay = await begin_turn(
        user_id=user_id,
        context_id=context.context_id,
        client_request_id=input_data.run_id,
        message=message,
    )
    conversation_id = str(conversation["conversation_id"])
    response_message_id = assistant_message_id(user_id, input_data.run_id)
    upstream: AsyncIterator[str] | None = None

    if replay is None:
        try:
            history = await load_model_history(conversation_id, user_id)
            upstream = await service.create_delta_stream(
                AskRinRequestContext(
                    action=LlmAction.ASK_RIN_CHAN,
                    question=context.question,
                    options=context.options,
                    user_question=message,
                    chat_history=history,
                    concept_name=context.concept_name,
                    bloom_level=context.bloom_level,
                    rag_context=getattr(context, "rag_context", ""),
                    question_image=context.question_image,
                    option_images=context.option_images,
                )
            )
        except Exception:
            await refund_turn(user_id=user_id, client_request_id=input_data.run_id)
            raise

    async def event_stream() -> AsyncIterator[str]:
        for event in _run_started_events(
            encoder=encoder,
            input_data=input_data,
            conversation_id=conversation_id,
            context_id=context.context_id,
            response_message_id=response_message_id,
        ):
            yield event

        if replay is not None:
            for event in _replay_events(
                encoder=encoder,
                input_data=input_data,
                conversation_id=conversation_id,
                response_message_id=response_message_id,
                replay=replay,
            ):
                yield event
            return

        if upstream is None:
            raise RuntimeError("Ask Rin-chan stream was not initialized")
        async for event in _fresh_run_events(
            encoder=encoder,
            input_data=input_data,
            user_id=user_id,
            conversation_id=conversation_id,
            response_message_id=response_message_id,
            upstream=upstream,
        ):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type=encoder.get_content_type(),
        headers=SSE_STREAM_HEADERS,
    )
