import asyncio

from api.core import mongo_store


class _SubjectProgressRepoStub:
    async def load_by_session_for_user(self, session_id: str, user_id: str):
        return {"session_id": session_id, "user_id": user_id, "source": "session"}

    async def load_for_user(self, job_id: str, user_id: str):
        return {"job_id": job_id, "user_id": user_id, "source": "job"}


def test_load_session_doc_for_user_uses_subject_progress_repo(monkeypatch):
    monkeypatch.setattr(mongo_store, "_subject_progress_repo", _SubjectProgressRepoStub())

    doc = asyncio.run(mongo_store.load_session_doc_for_user("sess-1", "user-1"))

    assert doc == {"session_id": "sess-1", "user_id": "user-1", "source": "session"}


def test_load_subject_progress_for_job_uses_subject_progress_repo(monkeypatch):
    monkeypatch.setattr(mongo_store, "_subject_progress_repo", _SubjectProgressRepoStub())

    doc = asyncio.run(mongo_store.load_subject_progress_for_job("job-1", "user-1"))

    assert doc == {"job_id": "job-1", "user_id": "user-1", "source": "job"}


def test_find_latest_session_for_job_remains_backward_compatible(monkeypatch):
    monkeypatch.setattr(mongo_store, "_subject_progress_repo", _SubjectProgressRepoStub())

    doc = asyncio.run(mongo_store.find_latest_session_for_job("job-1", "user-1"))

    assert doc == {"job_id": "job-1", "user_id": "user-1", "source": "job"}
