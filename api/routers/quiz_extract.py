"""
routers/quiz_extract.py — Quiz extraction endpoints via LLM.
"""

import os
import sys
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from fastapi import APIRouter, Form, HTTPException, Depends
from loguru import logger

from ..config import get_settings
from ..dependencies import get_current_user

# Setup sys path for content-processor so loaders can be imported
CONTENT_PROCESSOR_SRC = str(
    Path(__file__).resolve().parents[3] / "content-processor" / "src"
)
if CONTENT_PROCESSOR_SRC not in sys.path:
    sys.path.insert(0, CONTENT_PROCESSOR_SRC)

# S3 Client util from content_pipeline
from ..core.content_pipeline import get_s3_client

router = APIRouter(prefix="/api/quiz", tags=["quiz"])
MAX_PDF_BYTES = 50 * 1024 * 1024

EXTRACTION_PROMPT = """
You are given educational content that may include questions, explanations, and references to images or diagrams.

Your task is to extract or generate quiz questions (both single-choice and multiple-choice) from this content.

If the content is clearly a list of questions (e.g., a quiz, test, or practice worksheet), you MUST extract them and format them directly.

Important Instructions:

1. **DO NOT OMIT ANY TEXT.** Even if the text refers to or is adjacent to an image, you must extract the full surrounding question text exactly as it appears.
2. **IGNORE IMAGES ENTIRELY.** Do not describe, summarize, or attempt to interpret any image content. Only process the visible text — even if it partially depends on an image.
3. Every question must have EXACTLY 4 to 5 options. If there are fewer, you must logically create plausible distractors to reach at least 4.
4. Provide the correct answer using a zero-based index (`correctIndex` or an array of `correctIndexes`).

Return ONLY a JSON array in the following format. Ensure the JSON is valid.

[
    {
        "question": "What is the capital of France?",
        "type": "single",
        "options": ["London", "Berlin", "Paris", "Madrid"],
        "correctIndex": 2
    },
    {
        "question": "Which of the following are programming languages?",
        "type": "multiple",
        "options": ["JavaScript", "HTML", "Python", "CSS"],
        "correctIndexes": [0, 2]
    }
]

User constraint overrides (if any, follow these above defaults):
<<<USER_PROMPT>>>

Always return raw JSON list of question dictionaries with no markdown codeblocks unless wrapped around the full list.
"""


def _clean_json_response(content: str) -> list:
    """Extract list JSON from LLM response safely"""
    match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    
    # Try direct parse
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception:
        pass
        
    return []


def _build_s3_object_url(endpoint_url: str, bucket_name: str, object_key: str) -> str:
    endpoint = endpoint_url.rstrip("/")
    key = object_key.lstrip("/")
    return f"{endpoint}/{bucket_name}/{key}"


def _extract_llm_content(payload: dict) -> str:
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(text_parts)

    return ""


async def _invoke_pdf_extract_llm(
    *,
    pdf_url: str,
    prompt: str,
    model: str,
    base_url: str,
    api_key: Optional[str],
) -> list:
    api_base = (base_url or "").rstrip("/")
    if not api_base:
        raise RuntimeError("LLM base URL is not configured")

    # Keep compatibility with OpenAI-like gateways that require /v1 suffix.
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"

    endpoint = f"{api_base}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": pdf_url}},
                ],
            }
        ],
    }

    logger.info(
        "[quiz_extract] llm_request_start endpoint={} model={} pdf_url={} prompt_chars={}",
        endpoint,
        model,
        pdf_url,
        len(prompt),
    )

    llm_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
    llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)

    logger.info(
        "[quiz_extract] llm_request_done status_code={} duration_ms={}",
        response.status_code,
        llm_duration_ms,
    )

    response_payload = response.json()
    llm_text = _extract_llm_content(response_payload)
    return _clean_json_response(llm_text)


@router.post("/extract")
async def extract_quiz(
    s3_key: str = Form(...),
    user_prompt: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
):
    """Extract quiz questions from a PDF already uploaded to S3."""
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()

    logger.info(
        "[quiz_extract] request_start request_id={} user_id={} has_user_prompt={} s3_key_raw={}",
        request_id,
        user_id,
        bool(user_prompt and user_prompt.strip()),
        s3_key,
    )

    if not s3_key or not s3_key.strip():
        logger.warning(
            "[quiz_extract] validation_failed request_id={} reason=missing_s3_key",
            request_id,
        )
        raise HTTPException(status_code=400, detail="s3_key is required.")

    s3_client = get_s3_client()
    settings = get_settings()

    try:
        if not s3_client or not settings.s3_bucket_name:
            raise HTTPException(status_code=500, detail="S3 is not configured.")

        if not settings.s3_endpoint_url:
            raise HTTPException(status_code=500, detail="S3 endpoint URL is not configured.")

        if not settings.llm_base_url or not settings.llm_model:
            raise HTTPException(status_code=500, detail="LLM configuration is missing.")

        normalized_key = s3_key.strip().lstrip("/")
        required_prefix = f"uploads/quiz_extract/{user_id}/"
        if not normalized_key.startswith(required_prefix):
            logger.warning(
                "[quiz_extract] validation_failed request_id={} reason=forbidden_s3_key normalized_key={} required_prefix={}",
                request_id,
                normalized_key,
                required_prefix,
            )
            raise HTTPException(status_code=403, detail="Forbidden s3_key for current user.")

        try:
            object_size = int(
                s3_client.head_object(
                    Bucket=settings.s3_bucket_name,
                    Key=normalized_key,
                ).get("ContentLength", 0)
            )
        except Exception as e:
            logger.error(
                "[quiz_extract] s3_head_failed request_id={} key={} error={}",
                request_id,
                normalized_key,
                e,
            )
            raise HTTPException(status_code=400, detail="Unable to inspect PDF from s3_key.")

        logger.info(
            "[quiz_extract] s3_head_ok request_id={} key={} object_size_bytes={}",
            request_id,
            normalized_key,
            object_size,
        )

        if object_size <= 0:
            raise HTTPException(status_code=400, detail="Source PDF is empty.")
        if object_size > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Source PDF exceeds 50MB limit. Please split the file before upload.",
            )

        prompt = EXTRACTION_PROMPT.replace(
            "<<<USER_PROMPT>>>",
            user_prompt if user_prompt else "No additional constraints.",
        )

        pdf_url = _build_s3_object_url(
            settings.s3_endpoint_url,
            settings.s3_bucket_name,
            normalized_key,
        )

        try:
            questions = await _invoke_pdf_extract_llm(
                pdf_url=pdf_url,
                prompt=prompt,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )
        except httpx.HTTPStatusError as e:
            body = e.response.text if hasattr(e, "response") and e.response is not None else str(e)
            logger.error(
                "[quiz_extract] llm_http_error request_id={} key={} error_body={}",
                request_id,
                normalized_key,
                body,
            )
            raise HTTPException(status_code=502, detail="Quiz extraction request to LLM failed.")
        except Exception as e:
            logger.error(
                "[quiz_extract] llm_invoke_error request_id={} key={} error={}",
                request_id,
                normalized_key,
                e,
            )
            raise HTTPException(status_code=502, detail="Quiz extraction failed during LLM invocation.")

        if not questions:
            raise HTTPException(status_code=502, detail="No quiz questions extracted from PDF.")

        total_duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "[quiz_extract] request_success request_id={} key={} questions_count={} duration_ms={}",
            request_id,
            normalized_key,
            len(questions),
            total_duration_ms,
        )

        return {
            "success": True,
            "data": {
                "questions": questions,
                "s3_key": normalized_key,
                "failed_chunks": 0,
                "total_chunks": 1,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        total_duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.error(
            "[quiz_extract] request_failed request_id={} error={} duration_ms={}",
            request_id,
            str(e),
            total_duration_ms,
        )
        raise HTTPException(status_code=500, detail="Quiz extraction internal error")

