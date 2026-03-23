"""
schemas/exercise.py — Exercise-related Pydantic models.
"""

from typing import Optional, Dict, Any, List, Literal

from pydantic import BaseModel, Field, ConfigDict


class ExerciseOption(BaseModel):
    key: str
    value: str


class NextConceptResponse(BaseModel):
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: str
    step: int
    max_steps: int


class TheoryResponse(BaseModel):
    content: str
    examples: List[str]


class ExerciseResponse(BaseModel):
    exercise_id: str
    concept_name: str
    concept_idx: int
    bloom_level: int
    bloom_label: str
    exercise_type: str = "mcq"
    question: str
    options: Dict[str, str] = Field(default_factory=dict)
    statement: Optional[str] = None
    hint: Optional[str] = None
    items: List[str] = Field(default_factory=list)
    pairs: List[Dict[str, str]] = Field(default_factory=list)
    right_items: List[str] = Field(default_factory=list)
    rubric: List[str] = Field(default_factory=list)
    step: int
    max_steps: int
    theory: Optional[Dict[str, Any]] = None


class SubmitAnswerPayload(BaseModel):
    choice: Optional[str] = None
    choices: List[str] = Field(default_factory=list)
    boolean: Optional[bool] = None
    blanks: List[str] = Field(default_factory=list)
    ordering: List[str] = Field(default_factory=list)
    matching: Dict[str, str] = Field(default_factory=dict)
    text: Optional[str] = None


class SubmitAnswerRequest(BaseModel):
    answer: SubmitAnswerPayload


class SubmitAnswerResponse(BaseModel):
    is_correct: bool
    correct_option: str
    explanation: str
    concept_name: str
    bloom_level: int
    mastery_after: float
    avg_mastery: float
    step: int
    session_completed: bool
    stats: Dict[str, Any]


class TutorChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class TutorChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_question: str = Field(..., alias="userQuestion", min_length=1, max_length=1000)
    chat_history: List[TutorChatMessage] = Field(default_factory=list, alias="chatHistory")
    stream: bool = False


class TutorChatResponse(BaseModel):
    explanation: str
