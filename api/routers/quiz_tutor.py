"""
routers/quiz_tutor.py — Quiz ask-AI endpoints backed by shared LLM runtime.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.config import get_settings
from api.core.quiz.quiz_tutor import create_quiz_tutor_stream, generate_quiz_tutor_response
from api.core.shared.llm import SSE_STREAM_HEADERS
from api.dependencies import get_current_user
from api.exceptions import AppError
from api.rate_limit import is_admin_request, limiter
from api.schemas.common import StandardResponse
from api.schemas.quiz_tutor import QuizTutorRequest, QuizTutorResponseData

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


@router.post("/ask-ai", response_model=StandardResponse[QuizTutorResponseData])
@limiter.limit(get_settings().rate_limit_ask_ai, exempt_when=is_admin_request)
async def ask_ai_about_quiz(
    request: Request,
    req: QuizTutorRequest,
    user_id: Annotated[str, Depends(get_current_user)],
):
    """Ask an AI tutor for help understanding a quiz question (stream or single response)."""
    del request
    del user_id

    try:
        if req.stream:
            stream = await create_quiz_tutor_stream(
                question=req.question,
                options=req.options,
                user_question=req.user_question,
                chat_history=[item.model_dump() for item in req.chat_history],
                question_image=req.question_image,
                option_images=req.option_images,
            )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers=SSE_STREAM_HEADERS,
            )

        payload = await asyncio.to_thread(
            generate_quiz_tutor_response,
            question=req.question,
            options=req.options,
            user_question=req.user_question,
            chat_history=[item.model_dump() for item in req.chat_history],
            question_image=req.question_image,
            option_images=req.option_images,
        )
        data = QuizTutorResponseData.model_validate(payload["data"])
        return StandardResponse(data=data)
    except ValueError as exc:
        raise AppError(str(exc), status_code=400) from exc
    except RuntimeError as exc:
        raise AppError(str(exc), status_code=502) from exc
