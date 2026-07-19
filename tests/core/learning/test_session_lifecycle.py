import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.domains.learning.session import SessionManager
from api.exceptions import AppError


def _manager_shell() -> SessionManager:
    manager = object.__new__(SessionManager)
    manager._sessions = {}
    manager._recovery_locks = {}
    manager._subject_session_locks = {}
    manager._subject_session_ids = {}
    return manager


@pytest.mark.asyncio
async def test_load_session_history_propagates_read_failure_as_503(monkeypatch):
    manager = _manager_shell()

    async def boom(job_id: str, user_id: str):
        del job_id, user_id
        from pymongo.errors import ServerSelectionTimeoutError

        raise ServerSelectionTimeoutError("mongo unavailable")

    monkeypatch.setattr(
        "api.domains.learning.session.load_subject_progress_for_user",
        boom,
    )

    with pytest.raises(AppError) as exc_info:
        await manager._load_session_history("job-1", "user-1", None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.meta is not None
    assert exc_info.value.meta["retryable"] is True


@pytest.mark.asyncio
async def test_create_session_from_pipeline_db_read_failure_does_not_register(
    monkeypatch,
):
    """DB read failure at create boundary must not register or snapshot progress."""
    manager = _manager_shell()
    manager._device = "cpu"
    manager._saint_model = object()
    manager._q_net = object()
    manager._mastery_threshold = 0.7

    fake_env = SimpleNamespace()
    fake_env.reset = lambda seed=None: (None, {})
    fake_env.inject_history = lambda *a, **k: None
    fake_env.build_observation = lambda: None

    monkeypatch.setattr(
        "api.domains.learning.session.AdaptiveLearningEnv",
        lambda *a, **k: fake_env,
    )
    monkeypatch.setattr(manager, "_build_external_embeddings", lambda *a, **k: None)

    async def boom(job_id: str, user_id: str):
        del job_id, user_id
        from pymongo.errors import ServerSelectionTimeoutError

        raise ServerSelectionTimeoutError("mongo unavailable")

    monkeypatch.setattr(
        "api.domains.learning.session.load_subject_progress_for_user",
        boom,
    )
    persist_spy = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "persist_subject_progress", persist_spy)

    with pytest.raises(AppError) as exc_info:
        await manager.create_session_from_pipeline(
            concepts_data={"c1": {"name": "Concept One", "definition": "A concept."}},
            concept_map={"c1": 0},
            prereq_edges=[],
            job_id="job-1",
            user_id="user-1",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.meta is not None
    assert exc_info.value.meta["retryable"] is True
    persist_spy.assert_not_awaited()
    assert manager._sessions == {}


@pytest.mark.asyncio
async def test_load_session_history_missing_progress_is_empty(monkeypatch):
    manager = _manager_shell()

    async def missing(job_id: str, user_id: str):
        del job_id, user_id

    monkeypatch.setattr(
        "api.domains.learning.session.load_subject_progress_for_user",
        missing,
    )

    history, correct, answered = await manager._load_session_history("job-1", "user-1", None)
    assert history == []
    assert correct == 0
    assert answered == 0


@pytest.mark.asyncio
async def test_get_or_create_pipeline_session_serializes_concurrent_creation():
    manager = _manager_shell()
    create_calls = 0

    async def _fake_create(self, **kwargs):
        nonlocal create_calls
        create_calls += 1
        await asyncio.sleep(0)
        session = SimpleNamespace(
            session_id="session-1",
            user_id=kwargs["user_id"],
            job_id=kwargs["job_id"],
            status="active",
            accessed_at=0.0,
        )
        return self._register_session(session)

    manager.create_session_from_pipeline = MethodType(_fake_create, manager)
    job_doc = {
        "job_id": "job-1",
        "result": {
            "concepts_data": {},
            "concept_map": {"c1": 0},
            "prereq_edges": [],
        },
    }

    first, second = await asyncio.gather(
        manager.get_or_create_pipeline_session(
            job_doc=job_doc,
            subject_progress=None,
            user_id="user-1",
            max_steps=50,
        ),
        manager.get_or_create_pipeline_session(
            job_doc=job_doc,
            subject_progress=None,
            user_id="user-1",
            max_steps=50,
        ),
    )

    assert first == (first[0], True)
    assert second == (first[0], False)
    assert create_calls == 1


@pytest.mark.asyncio
async def test_get_or_create_pipeline_session_reuses_loaded_recovery_documents(monkeypatch):
    manager = _manager_shell()
    create_calls = 0

    async def _unexpected_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("recovery should reuse documents already loaded by the route")

    async def _fake_create(self, **kwargs):
        nonlocal create_calls
        create_calls += 1
        session = SimpleNamespace(
            session_id=kwargs["session_id"],
            user_id=kwargs["user_id"],
            job_id=kwargs["job_id"],
            status=kwargs["history_source_doc"]["status"],
            accessed_at=0.0,
        )
        return self._register_session(session)

    monkeypatch.setattr(
        "api.domains.learning.session.load_subject_progress_by_session_for_user",
        _unexpected_load,
    )
    monkeypatch.setattr("api.domains.learning.session.load_pipeline_job_for_user", _unexpected_load)
    manager.create_session_from_pipeline = MethodType(_fake_create, manager)

    session, created = await manager.get_or_create_pipeline_session(
        job_doc={
            "job_id": "job-1",
            "result": {
                "concepts_data": {},
                "concept_map": {"c1": 0},
                "prereq_edges": [],
            },
        },
        subject_progress={
            "job_id": "job-1",
            "last_session_id": "session-1",
            "status": "active",
            "exercise_history": [],
        },
        user_id="user-1",
        max_steps=50,
    )

    assert session.session_id == "session-1"
    assert created is False
    assert create_calls == 1


@pytest.mark.asyncio
async def test_recovery_programming_error_propagates_instead_of_becoming_missing(monkeypatch):
    manager = _manager_shell()
    monkeypatch.setattr(
        manager,
        "create_session_from_pipeline",
        AsyncMock(side_effect=ValueError("invalid persisted invariant")),
    )

    with pytest.raises(ValueError, match="invalid persisted invariant"):
        await manager._recover_session_from_documents(
            session_id="session-1",
            user_id="user-1",
            session_doc={"job_id": "job-1", "max_steps": 50},
            job_doc={
                "result": {
                    "concepts_data": {"c1": {}},
                    "concept_map": {"c1": 0},
                    "prereq_edges": [],
                }
            },
        )
