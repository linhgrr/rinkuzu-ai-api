"""Schemas for FastAPI-owned quiz draft processing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuizDraftCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    s3_key: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    file_size: int | None = Field(default=None, ge=1)
    category_id: str | None = None
    description: str | None = Field(default=None, max_length=1000)
    prompt: str | None = Field(default=None, max_length=2000)


class QuizDraftPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    category_id: str | None = None
    questions: list[dict] | None = None


class QuizDraftSubmitRequest(BaseModel):
    quiz_id: str = Field(min_length=1)


class QuizDraftPdfInfo(BaseModel):
    s3_key: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    page_count: int | None = None


class QuizDraftProgress(BaseModel):
    processed: int = 0
    total: int = 0
    percent: int = 0


class QuizDraftResponseData(BaseModel):
    draft_id: str | None = None
    title: str | None = None
    description: str | None = None
    category_id: str | None = None
    prompt: str | None = None
    pdf: QuizDraftPdfInfo = Field(default_factory=QuizDraftPdfInfo)
    status: str | None = None
    progress: QuizDraftProgress = Field(default_factory=QuizDraftProgress)
    questions: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    submitted_quiz_id: str | None = None
    created_at: Any = None
    updated_at: Any = None
    expires_at: Any = None


class QuizDraftSingleResponse(BaseModel):
    success: bool = True
    draft: QuizDraftResponseData


class QuizDraftListResponse(BaseModel):
    success: bool = True
    drafts: list[QuizDraftResponseData] = Field(default_factory=list)
