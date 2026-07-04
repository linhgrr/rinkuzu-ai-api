import asyncio
from types import MethodType, SimpleNamespace

import pytest

from api.core.learning.session import SessionManager


def _manager_shell() -> SessionManager:
    manager = object.__new__(SessionManager)
    manager._sessions = {}
    manager._recovery_locks = {}
    manager._subject_session_locks = {}
    manager._subject_session_ids = {}
    return manager


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
        "api.core.learning.session.load_subject_progress_by_session_for_user",
        _unexpected_load,
    )
    monkeypatch.setattr("api.core.learning.session.load_pipeline_job_for_user", _unexpected_load)
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
