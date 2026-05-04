"""Learning domain modules.

Keep heavy ML-backed services lazy so simple schema/domain imports do not require
torch to be installed or models to be loaded during test collection.
"""

from importlib import import_module

__all__ = ["ExerciseRecord", "ExerciseService", "SessionManager", "SessionState"]


def __getattr__(name: str):
    if name in {"ExerciseRecord", "SessionManager", "SessionState"}:
        session = import_module(".session", __name__)
        return getattr(session, name)
    if name == "ExerciseService":
        exercise_service = import_module(".exercise_service", __name__)
        return exercise_service.ExerciseService
    raise AttributeError(name)
