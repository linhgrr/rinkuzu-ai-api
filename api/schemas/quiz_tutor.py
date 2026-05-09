"""
schemas/quiz_tutor.py — Quiz ask-AI request/response models.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QuizTutorChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class QuizTutorRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, max_length=12000)
    options: list[str] = Field(..., min_length=2, max_length=8)
    user_question: str | None = Field(default=None, alias="userQuestion", max_length=1000)
    question_image: str | None = Field(default=None, alias="questionImage")
    option_images: list[str | None] = Field(default_factory=list, alias="optionImages")
    chat_history: list[QuizTutorChatMessage] = Field(default_factory=list, alias="chatHistory")
    stream: bool = False


class QuizTutorResponseData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    explanation: str
    structured: dict | None = None
    timestamp: str
    turn_count: int
