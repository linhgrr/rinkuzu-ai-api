from enum import StrEnum


class LearningSessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


class SubjectProgressStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


class SubjectHistoryStatus(StrEnum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED = "completed"


class ConceptStatus(StrEnum):
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"


class BloomLabel(StrEnum):
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


class ReadinessStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"


class PipelineSessionSource(StrEnum):
    NEW_SESSION = "new_session"
    EXISTING_SESSION = "existing_session"
