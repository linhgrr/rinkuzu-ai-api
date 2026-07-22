"""Unified Ask Rin-chan API."""

from __future__ import annotations

import json
from typing import Annotated, Any, NoReturn

from ag_ui.core import RunAgentInput  # noqa: TC002 - FastAPI resolves at runtime.
from fastapi import APIRouter, Depends, Request
from sse_starlette import EventSourceResponse

from api.dependencies import (
    get_chunk_chroma_store,
    get_current_user,
    get_session_manager,
    resolve_user_session,
)
from api.domains.learning.router import (
    _build_rag_context,
    _resolve_exercise_options,
    _resolve_exercise_question,
)
from api.exceptions import AppError
from api.schemas.common import StandardResponse, ok
from api.schemas.validators import (  # noqa: TC001 - FastAPI resolves at runtime.
    ExerciseContextPathID,
)
from api.shared.llm import SSE_STREAM_HEADERS, serialize_responses_sse_event
from api.shared.llm_usage import LlmAction

from .agui import (
    create_agui_response,
    latest_user_message,
    read_exercise_context_token,
    validate_run_identity,
)
from .context_tokens import (
    ExerciseContext,
    issue_context_token,
    quiz_context_id,
    read_context_token,
)
from .repository import (
    begin_turn,
    delete_conversation,
    finish_turn,
    get_conversation,
    load_model_history,
    refund_turn,
)
from .schemas import (
    AskRinChatRequest,
    AskRinConversationResponse,
    ExerciseContextResponse,
    RegisterExerciseContextRequest,
)
from .service import (
    AskRinImageUnsupportedError,
    AskRinRequestContext,
    get_ask_rin_chan_service,
)

router = APIRouter(prefix="/api/v1/ask-rin-chan", tags=["ask-rin-chan"])


def _raise_ask_rin_input_error(exc: ValueError) -> NoReturn:
    if isinstance(exc, AskRinImageUnsupportedError):
        raise AppError(
            code="ask_rin_image_unsupported",
            message="Image question is unavailable",
            detail=str(exc),
            status_code=422,
        ) from exc
    raise AppError(
        code="validation_error",
        message="Invalid Ask Rin-chan message",
        detail=str(exc),
        status_code=422,
    ) from exc


async def _create_stream_or_refund(
    *,
    service: Any,
    context: AskRinRequestContext,
    user_id: str,
    client_request_id: str,
) -> Any:
    try:
        return await service.create_stream(context)
    except ValueError as exc:
        await refund_turn(user_id=user_id, client_request_id=client_request_id)
        _raise_ask_rin_input_error(exc)


@router.post(
    "/contexts",
    response_model=StandardResponse[ExerciseContextResponse],
)
async def register_exercise_context(
    req: RegisterExerciseContextRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    manager: Any = Depends(get_session_manager),
) -> Any:
    if req.source == "adaptive":
        session = await resolve_user_session(manager, str(req.session_id), user_id)
        candidates = [session.current_exercise, *reversed(session.exercise_history)]
        exercise = next(
            (item for item in candidates if item and item.exercise_id == req.exercise_id),
            None,
        )
        if exercise is None:
            raise AppError(
                code="ask_rin_context_stale",
                message="Exercise context is stale",
                detail="Refresh the exercise and try again",
                status_code=409,
            )
        context_id = f"adaptive:{req.session_id}:{req.exercise_id}"
        context = ExerciseContext(
            context_id=context_id,
            user_id=user_id,
            question=_resolve_exercise_question(exercise),
            options=_resolve_exercise_options(exercise),
            concept_name=exercise.concept_name,
            bloom_level=exercise.bloom_level,
            session_id=req.session_id,
            exercise_id=req.exercise_id,
        )
        return ok(
            ExerciseContextResponse(
                exercise_context_id=context_id,
                exercise_context_token=issue_context_token(context),
            ).model_dump(by_alias=True)
        )

    options = [item.strip() for item in req.options if item.strip()]
    question = str(req.question).strip()
    context_id = quiz_context_id(question=question, options=options)
    context = ExerciseContext(
        context_id=context_id,
        user_id=user_id,
        question=question,
        options=options,
        concept_name=req.concept_name,
        bloom_level=req.bloom_level,
        question_image=str(req.question_image) if req.question_image else None,
        option_images=[str(item) if item else None for item in req.option_images],
    )
    return ok(
        ExerciseContextResponse(
            exercise_context_id=context_id,
            exercise_context_token=issue_context_token(context),
        ).model_dump(by_alias=True)
    )


async def _resolve_context(
    context: ExerciseContext,
    *,
    user_id: str,
    message: str,
    manager: Any,
    chunk_store: Any,
) -> ExerciseContext:
    if not context.session_id or not context.exercise_id:
        return context
    session = await resolve_user_session(manager, context.session_id, user_id)
    candidates = [session.current_exercise, *reversed(session.exercise_history)]
    exercise = next(
        (item for item in candidates if item and item.exercise_id == context.exercise_id),
        None,
    )
    if not exercise:
        raise AppError(
            code="ask_rin_context_stale",
            message="Exercise context is stale",
            detail="Refresh the exercise and try again",
            status_code=409,
        )
    rag_context = await _build_rag_context(session, chunk_store, message, k=3)
    return context.model_copy(
        update={
            "question": _resolve_exercise_question(exercise),
            "options": _resolve_exercise_options(exercise),
            "concept_name": exercise.concept_name,
            "bloom_level": exercise.bloom_level,
            "rag_context": rag_context,
        }
    )


@router.post("/chat")
async def ask_rin_chan(
    request: Request,
    req: AskRinChatRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    manager: Any = Depends(get_session_manager),
    chunk_store: Any = Depends(get_chunk_chroma_store),
) -> Any:
    del request
    context = read_context_token(req.exercise_context_token, user_id=user_id)
    context = await _resolve_context(
        context,
        user_id=user_id,
        message=req.message,
        manager=manager,
        chunk_store=chunk_store,
    )
    conversation, replay = await begin_turn(
        user_id=user_id,
        context_id=context.context_id,
        client_request_id=req.client_request_id,
        message=req.message.strip(),
    )
    conversation_id = str(conversation["conversation_id"])

    if replay is not None:

        async def replay_stream():
            if replay:
                yield serialize_responses_sse_event(
                    {"type": "response.output_text.delta", "delta": replay}
                )
            yield serialize_responses_sse_event(
                {
                    "type": "response.completed",
                    "response": {"conversationId": conversation_id, "replayed": True},
                }
            )

        return EventSourceResponse(replay_stream(), headers=SSE_STREAM_HEADERS, ping=15)

    history = await load_model_history(conversation_id, user_id)
    service = get_ask_rin_chan_service()
    upstream = await _create_stream_or_refund(
        service=service,
        context=AskRinRequestContext(
            action=LlmAction.ASK_RIN_CHAN,
            question=context.question,
            options=context.options,
            user_question=req.message,
            chat_history=history,
            concept_name=context.concept_name,
            bloom_level=context.bloom_level,
            rag_context=getattr(context, "rag_context", ""),
            question_image=context.question_image,
            option_images=context.option_images,
        ),
        user_id=user_id,
        client_request_id=req.client_request_id,
    )

    async def persisted_stream():
        full_response = ""
        completed = False
        try:
            async for chunk in upstream:
                text = chunk.decode(errors="replace")
                for line in text.splitlines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "response.output_text.delta":
                        full_response += str(event.get("delta") or "")
                    elif event.get("type") == "response.completed":
                        completed = True
                yield chunk
        finally:
            if full_response.strip():
                await finish_turn(
                    user_id=user_id,
                    client_request_id=req.client_request_id,
                    content=full_response.strip(),
                    interrupted=not completed,
                )
            else:
                await refund_turn(user_id=user_id, client_request_id=req.client_request_id)

    return EventSourceResponse(
        persisted_stream(),
        headers=SSE_STREAM_HEADERS,
        ping=15,
        send_timeout=30,
    )


@router.post("/agent")
async def run_ask_rin_chan_agent(
    request: Request,
    input_data: RunAgentInput,
    user_id: Annotated[str, Depends(get_current_user)],
    manager: Any = Depends(get_session_manager),
    chunk_store: Any = Depends(get_chunk_chroma_store),
) -> Any:
    """Run Ask Rin-chan using the official AG-UI input and event protocol."""
    try:
        context = read_context_token(read_exercise_context_token(input_data), user_id=user_id)
        validate_run_identity(input_data, context)
        message = latest_user_message(input_data)
    except (TypeError, ValueError) as exc:
        raise AppError(
            code="validation_error",
            message="Invalid AG-UI run input",
            detail=str(exc),
            status_code=422,
        ) from exc

    context = await _resolve_context(
        context,
        user_id=user_id,
        message=message,
        manager=manager,
        chunk_store=chunk_store,
    )
    try:
        return await create_agui_response(
            request_accept=request.headers.get("accept"),
            input_data=input_data,
            user_id=user_id,
            context=context,
            service=get_ask_rin_chan_service(),
        )
    except ValueError as exc:
        _raise_ask_rin_input_error(exc)


@router.get(
    "/conversations/{exercise_context_id}",
    response_model=StandardResponse[AskRinConversationResponse | None],
)
async def read_conversation(
    exercise_context_id: ExerciseContextPathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    result = await get_conversation(user_id, exercise_context_id)
    if result is None:
        return ok(None)
    conversation = result["conversation"]
    messages = result["messages"]
    return ok(
        {
            "conversationId": conversation.conversation_id,
            "exerciseContextId": conversation.exercise_context_id,
            "messages": [
                {
                    "messageId": item.message_id,
                    "role": item.role,
                    "content": item.content,
                    "status": item.status,
                    "createdAt": item.created_at.isoformat(),
                }
                for item in messages
            ],
        }
    )


@router.delete("/conversations/{exercise_context_id}")
async def clear_conversation(
    exercise_context_id: ExerciseContextPathID,
    user_id: Annotated[str, Depends(get_current_user)],
) -> Any:
    deleted = await delete_conversation(user_id, exercise_context_id)
    return ok({"deleted": deleted})
