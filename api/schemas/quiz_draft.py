"""Schemas for FastAPI-owned quiz draft processing."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves this at runtime for OpenAPI.
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuizDraftQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    question: str = Field(min_length=1)
    type: Literal["single", "multiple"]
    options: list[str] = Field(min_length=4, max_length=5)
    correct_index: int | None = Field(default=None, alias="correctIndex")
    correct_indexes: list[int] = Field(default_factory=list, alias="correctIndexes")

    @model_validator(mode="after")
    def validate_answer_shape(self) -> QuizDraftQuestion:
        option_count = len(self.options)
        if self.type == "single":
            if self.correct_index is None:
                raise ValueError("single-choice questions require correctIndex")
            if self.correct_indexes:
                raise ValueError("single-choice questions must not include correctIndexes")
            if not 0 <= self.correct_index < option_count:
                raise ValueError("correctIndex is out of range")
            return self

        if self.correct_index is not None:
            raise ValueError("multiple-choice questions must not include correctIndex")
        if not self.correct_indexes:
            raise ValueError("multiple-choice questions require correctIndexes")
        if len(set(self.correct_indexes)) != len(self.correct_indexes):
            raise ValueError("correctIndexes must be unique")
        if any(index < 0 or index >= option_count for index in self.correct_indexes):
            raise ValueError("correctIndexes contains an out-of-range value")
        return self


class QuizDraftCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200, description="Quiz title.")
    s3_key: str = Field(min_length=1, description="S3 object key for the source document.")
    file_name: str = Field(min_length=1, description="Original filename of the uploaded document.")
    file_size: int | None = Field(default=None, ge=1, description="File size in bytes.")
    category_id: str | None = Field(default=None, description="Optional category identifier.")
    description: str | None = Field(
        default=None, max_length=1000, description="Optional quiz description."
    )
    prompt: str | None = Field(
        default=None, max_length=2000, description="Custom AI prompt to guide question generation."
    )


class QuizDraftPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    category_id: str | None = None
    questions: list[QuizDraftQuestion] | None = None


class QuizDraftSubmitRequest(BaseModel):
    quiz_id: str = Field(min_length=1)


class QuizDraftPdfInfo(BaseModel):
    s3_key: str | None
    file_name: str | None
    file_size: int | None
    page_count: int | None


class QuizDraftProgress(BaseModel):
    processed: int = 0
    total: int = 0
    percent: int = 0


class QuizDraftResponseData(BaseModel):
    draft_id: str
    title: str
    description: str
    category_id: str | None
    prompt: str | None
    pdf: QuizDraftPdfInfo
    status: Literal[
        "queued",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "submitted",
        "expired",
    ]
    progress: QuizDraftProgress
    questions: list[QuizDraftQuestion]
    error: str | None
    submitted_quiz_id: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class QuizDraftSingleResponse(BaseModel):
    draft: QuizDraftResponseData


class QuizDraftListResponse(BaseModel):
    drafts: list[QuizDraftResponseData]
