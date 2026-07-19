"""Regression: cancel_requested is monotonic under concurrent/stale worker saves."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.shared.persistence import pipeline_jobs as store


class _EqField:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: object) -> bool:
        return (self.name, "==", other)

    def __hash__(self) -> int:
        return hash(self.name)


def _apply_update_ops(storage: dict[str, Any], ops: dict[str, Any]) -> None:
    """Apply the real Mongo-shaped contract: $set then $max (BSON false < true)."""
    storage.update(ops.get("$set", {}))
    for key, value in ops.get("$max", {}).items():
        current = storage.get(key)
        if current is None or value > current:
            storage[key] = value


def _filter_matches(storage: dict[str, Any] | None, args: tuple[Any, ...]) -> bool:
    if not storage:
        return False
    for arg in args:
        if isinstance(arg, dict):
            for key, expected in arg.items():
                actual = storage.get(key)
                if isinstance(expected, dict):
                    if "$in" in expected and actual not in expected["$in"]:
                        return False
                    if "$ne" in expected and actual == expected["$ne"]:
                        return False
                elif actual != expected:
                    return False
        elif (
            isinstance(arg, tuple)
            and len(arg) == 3
            and arg[1] == "=="
            and storage.get(arg[0]) != arg[2]
        ):
            return False
    return True


class _FakeFind:
    def __init__(
        self,
        storage: dict[str, Any] | None,
        update_calls: list[dict[str, Any]],
        *,
        matched: bool,
    ):
        self._storage = storage
        self._update_calls = update_calls
        self._matched = matched

    async def update(self, ops: dict[str, Any], **_kwargs: Any) -> SimpleNamespace:
        self._update_calls.append(ops)
        if not self._matched or self._storage is None:
            return SimpleNamespace(matched_count=0)
        _apply_update_ops(self._storage, ops)
        return SimpleNamespace(matched_count=1)

    def __await__(self):
        async def _resolve() -> SimpleNamespace | None:
            if not self._matched or self._storage is None:
                return None
            return SimpleNamespace(**self._storage)

        return _resolve().__await__()


class _FakePipelineJobDocument:
    job_id = _EqField("job_id")
    storage: ClassVar[dict[str, Any]] = {}
    update_calls: ClassVar[list[dict[str, Any]]] = []
    insert_calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any):
        self._fields = dict(kwargs)

    @classmethod
    def find_one(cls, *args: Any, **_kwargs: Any) -> _FakeFind:
        matched = _filter_matches(cls.storage, args)
        return _FakeFind(cls.storage or None, cls.update_calls, matched=matched)

    async def insert(self) -> None:
        self.insert_calls.append(dict(self._fields))
        type(self).storage = dict(self._fields)


@pytest.fixture
def fake_doc(monkeypatch: pytest.MonkeyPatch) -> type[_FakePipelineJobDocument]:
    monkeypatch.setattr(store, "PipelineJobDocument", _FakePipelineJobDocument)
    _FakePipelineJobDocument.storage = {}
    _FakePipelineJobDocument.update_calls = []
    _FakePipelineJobDocument.insert_calls = []
    return _FakePipelineJobDocument


def test_pipeline_job_update_ops_uses_max_not_set_for_cancel():
    ops = store._pipeline_job_update_ops(
        {
            "job_id": "j1",
            "progress": 10.0,
            "cancel_requested": False,
        }
    )
    assert "cancel_requested" not in ops["$set"]
    assert ops["$set"]["progress"] == 10.0
    assert ops["$max"] == {"cancel_requested": False}

    ops_true = store._pipeline_job_update_ops(
        {"job_id": "j1", "progress": 1.0, "cancel_requested": True}
    )
    assert ops_true["$max"] == {"cancel_requested": True}
    assert "cancel_requested" not in ops_true["$set"]


def test_max_merge_contract_false_cannot_overwrite_true():
    storage: dict[str, Any] = {"cancel_requested": True, "progress": 5.0}
    _apply_update_ops(
        storage,
        store._pipeline_job_update_ops(
            {"job_id": "j1", "progress": 50.0, "cancel_requested": False}
        ),
    )
    assert storage["cancel_requested"] is True
    assert storage["progress"] == 50.0


@pytest.mark.asyncio
async def test_stale_worker_save_false_after_cancel_true_keeps_true(fake_doc):
    """Real order: cancellation commits true, then stale worker save with false runs."""
    cancel_job = PipelineJob(job_id="job-1", filename="a.pdf", subject_id="s1")
    cancel_job.cancel_requested = True
    cancel_job.progress = 12.0
    cancel_job.status = PipelineStatus.EXTRACTING
    cancel_job.retry_count = 0

    fake_doc.storage = {
        "job_id": "job-1",
        "filename": "a.pdf",
        "subject_id": "s1",
        "progress": 12.0,
        "cancel_requested": True,
        "status": PipelineStatus.EXTRACTING.value,
        "retry_count": 0,
    }

    ok = await store.save_pipeline_job(cancel_job)
    assert ok is store.SaveJobOutcome.CANCEL_REQUESTED
    # Progress with cancel_requested already true is rejected (cancel wins).
    assert fake_doc.storage["cancel_requested"] is True
    assert fake_doc.storage["progress"] == 12.0

    stale_worker = PipelineJob(job_id="job-1", filename="a.pdf", subject_id="s1")
    stale_worker.cancel_requested = False
    stale_worker.progress = 88.0
    stale_worker.current_step = "extract"
    stale_worker.status = PipelineStatus.EXTRACTING
    stale_worker.retry_count = 0

    ok = await store.save_pipeline_job(stale_worker)
    assert ok is store.SaveJobOutcome.CANCEL_REQUESTED
    assert fake_doc.storage["cancel_requested"] is True
    # Progress must not advance under cancel_requested=true.
    assert fake_doc.storage["progress"] == 12.0
    assert fake_doc.storage.get("current_step") != "extract"
