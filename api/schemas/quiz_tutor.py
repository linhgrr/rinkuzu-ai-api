"""
schemas/quiz_tutor.py — Quiz ask-AI request/response models.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class QuizTutorChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class QuizTutorRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, max_length=12000)
    options: List[str] = Field(..., min_length=2, max_length=8)
    user_question: str = Field(..., alias="userQuestion", min_length=1, max_length=1000)
    question_image: Optional[str] = Field(default=None, alias="questionImage")
    option_images: List[Optional[str]] = Field(default_factory=list, alias="optionImages")
    chat_history: List[QuizTutorChatMessage] = Field(default_factory=list, alias="chatHistory")
    stream: bool = False


class QuizTutorResponseData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    explanation: str
    structured: Optional[dict] = None
    timestamp: str
    turn_count: int = Field(alias="turnCount")


class QuizTutorResponse(BaseModel):
    success: bool = True
    data: QuizTutorResponseData
