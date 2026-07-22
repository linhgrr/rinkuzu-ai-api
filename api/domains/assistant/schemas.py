"""HTTP contracts for Ask Rin-chan."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class RegisterExerciseContextRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source: Literal["quiz", "adaptive"] = "quiz"
    question: str | None = Field(default=None, min_length=1, max_length=12_000)
    options: list[str] = Field(default_factory=list, max_length=20)
    question_image: HttpUrl | None = Field(default=None, alias="questionImage")
    option_images: list[HttpUrl | None] = Field(default_factory=list, alias="optionImages")
    concept_name: str | None = Field(default=None, alias="conceptName", max_length=500)
    bloom_level: int | None = Field(default=None, alias="bloomLevel", ge=1, le=6)
    session_id: str | None = Field(default=None, alias="sessionId", min_length=1, max_length=160)
    exercise_id: str | None = Field(default=None, alias="exerciseId", min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_source_fields(self) -> RegisterExerciseContextRequest:
        if self.source == "adaptive":
            if not self.session_id or not self.exercise_id:
                raise ValueError("sessionId and exerciseId are required for adaptive context")
        elif not self.question or not self.question.strip():
            raise ValueError("question is required for quiz context")
        return self


class ExerciseContextResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exercise_context_id: str = Field(alias="exerciseContextId")
    exercise_context_token: str = Field(alias="exerciseContextToken")


class AskRinChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    exercise_context_token: str = Field(
        alias="exerciseContextToken", min_length=32, max_length=32_768
    )
    message: str = Field(min_length=1, max_length=1_000)
    client_request_id: str = Field(alias="clientRequestId", min_length=8, max_length=128)


class AskRinMessageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: str = Field(alias="messageId")
    role: Literal["user", "assistant"]
    content: str
    status: Literal["complete", "interrupted"]
    created_at: str = Field(alias="createdAt")


class AskRinConversationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: str = Field(alias="conversationId")
    exercise_context_id: str = Field(alias="exerciseContextId")
    messages: list[AskRinMessageResponse]
