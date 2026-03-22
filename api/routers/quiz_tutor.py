"""
routers/quiz_tutor.py — Quiz ask-AI endpoints backed by shared LLM runtime.
"""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.quiz_tutor import create_quiz_tutor_stream, generate_quiz_tutor_response
from ..dependencies import get_current_user
from ..schemas.quiz_tutor import QuizTutorRequest, QuizTutorResponse

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


@router.post("/ask-ai", response_model=QuizTutorResponse)
async def ask_ai_about_quiz(
    req: QuizTutorRequest,
    user_id: str = Depends(get_current_user),
):
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
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
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
        return QuizTutorResponse.model_validate(payload)
    except ValueError as exc:
        return JSONResponse(
            {"success": False, "error": str(exc)},
            status_code=400,
        )
    except RuntimeError as exc:
        return JSONResponse(
            {"success": False, "error": str(exc)},
            status_code=502,
        )
