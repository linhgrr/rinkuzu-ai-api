"""Schemas for FastAPI-owned quiz draft processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from datetime import datetime


class QuizDraftCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200, description="Quiz title.")
    s3_key: str = Field(min_length=1, description="S3 object key for the source document.")
    file_name: str = Field(min_length=1, description="Original filename of the uploaded document.")
    file_size: int | None = Field(default=None, ge=1, description="File size in bytes.")
    category_id: str | None = Field(default=None, description="Optional category identifier.")
    description: str | None = Field(default=None, max_length=1000, description="Optional quiz description.")
    prompt: str | None = Field(default=None, max_length=2000, description="Custom AI prompt to guide question generation.")


class QuizDraftPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    category_id: str | None = None
    questions: list[dict[str, object]] | None = None


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
    questions: list[dict[str, object]] = Field(default_factory=list)
    error: str | None = None
    submitted_quiz_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None


class QuizDraftSingleResponse(BaseModel):
    success: bool = True
    draft: QuizDraftResponseData


class QuizDraftListResponse(BaseModel):
    success: bool = True
    drafts: list[QuizDraftResponseData] = Field(default_factory=list)
