"""Learning domain modules."""

__all__ = ["ExerciseRecord", "ExerciseService", "SessionManager", "SessionState"]


def __getattr__(name: str):
    if name == "ExerciseService":
        from .exercise_service import ExerciseService

        return ExerciseService
    if name in {"ExerciseRecord", "SessionManager", "SessionState"}:
        from .session import ExerciseRecord, SessionManager, SessionState

        return {
            "ExerciseRecord": ExerciseRecord,
            "SessionManager": SessionManager,
            "SessionState": SessionState,
        }[name]
    raise AttributeError(name)
