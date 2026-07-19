"""State-machine CAS tests for pipeline cancel / retry / save monotonicity.

These assert Mongo-shaped filters and operators against a fake Beanie surface.
They do not claim live multi-writer concurrency against a real Mongo cluster.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
import pytest

from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.shared.persistence import pipeline_jobs as store
from api.shared.persistence.common import utc_now


class _EqField:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: object) -> bool:
        return (self.name, "==", other)

    def __hash__(self) -> int:
        return hash(self.name)


def _match_filter(doc: dict[str, Any], query: dict[str, Any] | tuple | object) -> bool:
    """Minimal matcher for dict filters and Beanie equality tuples."""
    if isinstance(query, tuple) and len(query) == 3 and query[1] == "==":
        return doc.get(query[0]) == query[2]
    if not isinstance(query, dict):
        return True
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$lt" in expected and not (actual is not None and actual < expected["$lt"]):
                return False
            if (
                "$type" in expected
                and expected["$type"] == "string"
                and not isinstance(actual, str)
            ):
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$gte" in expected:
                continue  # time windows not needed in these unit fakes
        elif actual != expected:
            return False
    return True


def _apply_ops(storage: dict[str, Any], ops: dict[str, Any]) -> None:
    if "$set" in ops:
        storage.update(ops["$set"])
    if "$max" in ops:
        for key, value in ops["$max"].items():
            current = storage.get(key)
            if current is None or value > current:
                storage[key] = value
    if "$inc" in ops:
        for key, value in ops["$inc"].items():
            storage[key] = int(storage.get(key) or 0) + int(value)


class _FakeFind:
    def __init__(
        self,
        storage: dict[str, Any] | None,
        *,
        matched: bool,
        update_calls: list[dict[str, Any]],
        find_calls: list[Any],
        query: Any,
    ):
        self._storage = storage
        self._matched = matched
        self._update_calls = update_calls
        self._find_calls = find_calls
        self._query = query

    async def update(self, ops: dict[str, Any], **kwargs: Any) -> Any:
        self._update_calls.append({"ops": ops, "kwargs": kwargs, "query": self._query})
        if not self._matched or self._storage is None:
            response_type = kwargs.get("response_type")
            if response_type is not None:
                return None
            return SimpleNamespace(matched_count=0)
        _apply_ops(self._storage, ops)
        response_type = kwargs.get("response_type")
        if response_type is not None:
            return SimpleNamespace(**self._storage)
        return SimpleNamespace(matched_count=1)


class _FakePipelineJobDocument:
    job_id = _EqField("job_id")
    user_id = _EqField("user_id")
    storage: ClassVar[dict[str, Any] | None] = None
    update_calls: ClassVar[list[dict[str, Any]]] = []
    find_calls: ClassVar[list[Any]] = []
    insert_calls: ClassVar[list[dict[str, Any]]] = []
    raise_on_find: ClassVar[BaseException | None] = None
    raise_on_update: ClassVar[BaseException | None] = None
    force_duplicate_on_insert: ClassVar[bool] = False

    def __init__(self, **kwargs: Any):
        self._fields = dict(kwargs)

    @classmethod
    def _resolve_doc(cls, *, matched: bool, kwargs: dict[str, Any]) -> Any:
        if not matched or cls.storage is None:
            return None
        proj = kwargs.get("projection_model")
        if proj is None:
            return SimpleNamespace(**cls.storage)
        fields = {
            name: cls.storage.get(name)
            for name in getattr(proj, "model_fields", {})
            if name in cls.storage
        }
        fields.setdefault("job_id", cls.storage.get("job_id", "job-1"))
        fields.setdefault("filename", cls.storage.get("filename", "a.pdf"))
        fields.setdefault("subject_id", cls.storage.get("subject_id", "s1"))
        fields.setdefault("status", cls.storage.get("status", "extracting"))
        try:
            return proj(**fields)
        except Exception:
            return SimpleNamespace(**cls.storage)

    @classmethod
    def find_one(cls, *args: Any, **kwargs: Any) -> _FakeFind | Any:
        if cls.raise_on_find is not None:
            raise cls.raise_on_find
        cls.find_calls.append({"args": args, "kwargs": kwargs})
        query: dict[str, Any] = {}
        for arg in args:
            if isinstance(arg, dict):
                query.update(arg)
            elif isinstance(arg, tuple) and len(arg) == 3 and arg[1] == "==":
                query[arg[0]] = arg[2]
        matched = cls.storage is not None and _match_filter(cls.storage, query)

        class _AwaitableDoc:
            def __await__(self):
                async def _resolve():
                    if cls.raise_on_find is not None:
                        raise cls.raise_on_find
                    return cls._resolve_doc(matched=matched, kwargs=kwargs)

                return _resolve().__await__()

            async def update(self, ops: dict[str, Any], **ukw: Any) -> Any:
                if cls.raise_on_update is not None:
                    raise cls.raise_on_update
                return await _FakeFind(
                    cls.storage if matched else None,
                    matched=matched,
                    update_calls=cls.update_calls,
                    find_calls=cls.find_calls,
                    query=query,
                ).update(ops, **ukw)

        return _AwaitableDoc()

    async def insert(self) -> None:
        if type(self).force_duplicate_on_insert:
            raise DuplicateKeyError("E11000 duplicate key")
        type(self).insert_calls.append(dict(self._fields))
        type(self).storage = dict(self._fields)


@pytest.fixture
def fake_doc(monkeypatch: pytest.MonkeyPatch) -> type[_FakePipelineJobDocument]:
    monkeypatch.setattr(store, "PipelineJobDocument", _FakePipelineJobDocument)
    _FakePipelineJobDocument.storage = None
    _FakePipelineJobDocument.update_calls = []
    _FakePipelineJobDocument.find_calls = []
    _FakePipelineJobDocument.insert_calls = []
    _FakePipelineJobDocument.raise_on_find = None
    _FakePipelineJobDocument.raise_on_update = None
    _FakePipelineJobDocument.force_duplicate_on_insert = False
    return _FakePipelineJobDocument


def _seed(fake_doc: type[_FakePipelineJobDocument], **fields: Any) -> dict[str, Any]:
    base = {
        "job_id": "job-1",
        "filename": "a.pdf",
        "subject_id": "s1",
        "user_id": "user-1",
        "status": PipelineStatus.EXTRACTING.value,
        "retry_count": 0,
        "cancel_requested": False,
        "retryable": False,
        "progress": 0.2,
        "source_s3_key": "uploads/a.pdf",
        "current_step": "extract",
        "total_chunks": 0,
        "total_pages": 0,
        "page_batch_size": 10,
        "batch_count": 0,
        "failed_batch_count": 0,
        "partial_success": False,
        "concepts_extracted": 0,
        "concepts_after_merge": 0,
        "relations_verified": 0,
        "graph_stats": {},
        "quality_report": None,
        "debug_trace": [],
        "result": None,
        "partial_graph": None,
        "error_message": None,
        "error_code": None,
        "user_message": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "heartbeat_at": utc_now(),
        "completed_at": None,
        "prs_threshold": None,
        "min_confidence": 0.6,
        "apply_reduction": True,
        "eta_seconds": None,
    }
    base.update(fields)
    fake_doc.storage = base
    return base


@pytest.mark.asyncio
async def test_cancel_vs_progress_keeps_cancel_true(fake_doc):
    storage = _seed(fake_doc, cancel_requested=False, progress=0.2)
    result = await store.request_cancel_pipeline_job_for_user("job-1", "user-1")
    assert result.outcome is store.CancelJobOutcome.REQUESTED
    assert storage["cancel_requested"] is True

    worker = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.EXTRACTING,
        retry_count=0,
        cancel_requested=False,
        progress=0.8,
    )
    ok = await store.save_pipeline_job(worker)
    assert ok is store.SaveJobOutcome.CANCEL_REQUESTED
    assert storage["cancel_requested"] is True
    # Cancel wins: progress write must not apply when flag is already true.
    assert storage["progress"] != 0.8


@pytest.mark.asyncio
async def test_cancel_vs_complete_terminal_not_overwritten(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.COMPLETED.value,
        cancel_requested=False,
        progress=1.0,
    )
    result = await store.request_cancel_pipeline_job_for_user("job-1", "user-1")
    assert result.outcome is store.CancelJobOutcome.ALREADY_TERMINAL
    assert result.status == PipelineStatus.COMPLETED.value
    assert storage["status"] == PipelineStatus.COMPLETED.value
    assert storage["cancel_requested"] is False

    stale = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.EXTRACTING,
        retry_count=0,
        progress=0.5,
    )
    ok = await store.save_pipeline_job(stale)
    assert ok is store.SaveJobOutcome.ALREADY_TERMINAL
    assert storage["status"] == PipelineStatus.COMPLETED.value
    assert storage["progress"] == 1.0


@pytest.mark.asyncio
async def test_completion_after_cancel_is_rejected_not_completed_with_flag(fake_doc):
    """Cancel wins finalization: COMPLETED must not commit when cancel_requested=true."""
    storage = _seed(fake_doc, status=PipelineStatus.EXTRACTING.value)
    cancel = await store.request_cancel_pipeline_job_for_user("job-1", "user-1")
    assert cancel.outcome is store.CancelJobOutcome.REQUESTED

    completing_worker = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.COMPLETED,
        retry_count=0,
        cancel_requested=False,
        progress=1.0,
    )
    outcome = await store.save_pipeline_job(completing_worker)
    assert outcome is store.SaveJobOutcome.CANCEL_REQUESTED
    assert storage["status"] == PipelineStatus.EXTRACTING.value
    assert storage["cancel_requested"] is True
    assert storage["progress"] != 1.0


@pytest.mark.asyncio
async def test_cancelled_terminal_write_allowed_when_flag_true(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.EXTRACTING.value,
        cancel_requested=True,
        progress=0.5,
    )
    cancelling = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.CANCELLED,
        retry_count=0,
        cancel_requested=True,
        progress=0.5,
        retryable=True,
        error_code="pipeline_cancelled",
    )
    outcome = await store.save_pipeline_job(cancelling)
    assert outcome is store.SaveJobOutcome.APPLIED
    assert storage["status"] == PipelineStatus.CANCELLED.value
    assert storage["cancel_requested"] is True


@pytest.mark.asyncio
async def test_double_cancel_is_idempotent(fake_doc):
    _seed(fake_doc)
    first = await store.request_cancel_pipeline_job_for_user("job-1", "user-1")
    second = await store.request_cancel_pipeline_job_for_user("job-1", "user-1")
    assert first.outcome is store.CancelJobOutcome.REQUESTED
    assert second.outcome is store.CancelJobOutcome.REQUESTED
    assert fake_doc.storage is not None
    assert fake_doc.storage["cancel_requested"] is True


@pytest.mark.asyncio
async def test_retry_clears_cancel_only_via_authorized_transition(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.CANCELLED.value,
        retryable=True,
        cancel_requested=True,
        retry_count=0,
        source_s3_key="uploads/a.pdf",
    )
    # Unauthorized progress/save cannot clear cancel.
    worker = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.QUEUED,
        retry_count=0,
        cancel_requested=False,
    )
    outcome = await store.save_pipeline_job(worker)
    assert outcome is store.SaveJobOutcome.ALREADY_TERMINAL
    # Terminal doc: save is rejected; cancel stays true if still present.
    assert storage["cancel_requested"] is True

    result = await store.transition_pipeline_job_for_retry("job-1", "user-1", max_retry_count=3)
    assert result.outcome is store.RetryJobOutcome.RETRIED
    assert storage["cancel_requested"] is False
    assert storage["status"] == PipelineStatus.QUEUED.value
    assert storage["retry_count"] == 1
    # Query shape includes owner + terminal-retryable predicates + max retries.
    cancel_clear_ops = [call for call in fake_doc.update_calls if "$inc" in call["ops"]]
    assert cancel_clear_ops
    assert cancel_clear_ops[-1]["ops"]["$set"]["cancel_requested"] is False
    assert cancel_clear_ops[-1]["ops"]["$inc"] == {"retry_count": 1}


@pytest.mark.asyncio
async def test_retry_vs_stale_worker_generation_cas(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.FAILED.value,
        retryable=True,
        retry_count=0,
        cancel_requested=True,
        source_s3_key="uploads/a.pdf",
    )
    result = await store.transition_pipeline_job_for_retry("job-1", "user-1", max_retry_count=3)
    assert result.outcome is store.RetryJobOutcome.RETRIED
    assert storage["retry_count"] == 1
    assert storage["status"] == PipelineStatus.QUEUED.value

    stale_worker = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.EXTRACTING,
        retry_count=0,  # old generation
        progress=0.9,
        cancel_requested=False,
    )
    ok = await store.save_pipeline_job(stale_worker)
    assert ok is store.SaveJobOutcome.STALE_GENERATION
    # Generation CAS: stale retry_count=0 cannot overwrite post-retry document.
    assert storage["retry_count"] == 1
    assert storage["status"] == PipelineStatus.QUEUED.value
    assert storage["progress"] != 0.9


@pytest.mark.asyncio
async def test_load_for_user_db_outage_propagates(fake_doc):
    fake_doc.raise_on_find = ServerSelectionTimeoutError("mongo down")
    with pytest.raises(ServerSelectionTimeoutError):
        await store.load_pipeline_job_for_user("job-1", "user-1")


@pytest.mark.asyncio
async def test_load_cancel_flag_db_outage_propagates(fake_doc):
    fake_doc.raise_on_find = ServerSelectionTimeoutError("mongo down")
    with pytest.raises(ServerSelectionTimeoutError):
        await store.load_pipeline_job_cancel_requested("job-1")


@pytest.mark.asyncio
async def test_create_collision_never_replays_incoming_payload_over_winner(fake_doc):
    fake_doc.storage = None
    original_insert = _FakePipelineJobDocument.insert

    async def insert_then_seed(self: _FakePipelineJobDocument) -> None:
        type(self).storage = {
            "job_id": "job-1",
            "status": PipelineStatus.PENDING.value,
            "retry_count": 0,
            "cancel_requested": False,
            "progress": 0.0,
            "user_id": "winner-user",
            "filename": "winner.pdf",
            "subject_id": "winner-subject",
            "source_s3_key": "winner/source.pdf",
        }
        raise DuplicateKeyError("E11000")

    monkey_attr = insert_then_seed
    _FakePipelineJobDocument.insert = monkey_attr  # type: ignore[method-assign]

    job = PipelineJob(
        job_id="job-1",
        filename="a.pdf",
        subject_id="s1",
        user_id="user-1",
        status=PipelineStatus.QUEUED,
        retry_count=0,
        progress=0.01,
    )
    try:
        job.source_s3_key = "incoming/source.pdf"
        outcome = await store.create_pipeline_job(job)
    finally:
        _FakePipelineJobDocument.insert = original_insert  # type: ignore[method-assign]
    assert outcome is store.CreateJobOutcome.COLLISION
    assert fake_doc.storage is not None
    assert fake_doc.storage["user_id"] == "winner-user"
    assert fake_doc.storage["source_s3_key"] == "winner/source.pdf"
    assert fake_doc.storage["progress"] == 0.0
    assert fake_doc.update_calls == []


@pytest.mark.asyncio
async def test_worker_save_never_inserts_missing_job(fake_doc):
    fake_doc.storage = None
    job = PipelineJob(job_id="missing", filename="a.pdf", subject_id="s1")
    outcome = await store.save_pipeline_job(job)
    assert outcome is store.SaveJobOutcome.STALE_GENERATION
    assert fake_doc.insert_calls == []


@pytest.mark.asyncio
async def test_cancel_not_found(fake_doc):
    fake_doc.storage = None
    result = await store.request_cancel_pipeline_job_for_user("missing", "user-1")
    assert result.outcome is store.CancelJobOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_cancel_owned_nonterminal_race_is_conflict_not_false_404(monkeypatch):
    class _AwaitableRow:
        def __init__(self, row: Any):
            self._row = row

        def __await__(self):
            async def _resolve():
                return self._row

            return _resolve().__await__()

    class _MissedUpdate:
        async def update(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    responses = iter(
        [
            _MissedUpdate(),
            _AwaitableRow(SimpleNamespace(status=PipelineStatus.QUEUED)),
            _AwaitableRow(SimpleNamespace(cancel_requested=False)),
        ]
    )

    class _RaceDocument:
        job_id = _EqField("job_id")
        user_id = _EqField("user_id")

        @classmethod
        def find_one(cls, *_args: Any, **_kwargs: Any) -> Any:
            return next(responses)

    monkeypatch.setattr(store, "PipelineJobDocument", _RaceDocument)
    result = await store.request_cancel_pipeline_job_for_user("job-1", "user-1")
    assert result.outcome is store.CancelJobOutcome.CONFLICT
    assert result.status == PipelineStatus.QUEUED.value


@pytest.mark.asyncio
async def test_retry_invalid_state_and_max(fake_doc):
    _seed(fake_doc, status=PipelineStatus.EXTRACTING.value, retryable=True)
    result = await store.transition_pipeline_job_for_retry("job-1", "user-1", max_retry_count=3)
    assert result.outcome is store.RetryJobOutcome.INVALID_STATE

    _seed(
        fake_doc,
        status=PipelineStatus.FAILED.value,
        retryable=True,
        retry_count=3,
        source_s3_key="k",
    )
    result = await store.transition_pipeline_job_for_retry("job-1", "user-1", max_retry_count=3)
    assert result.outcome is store.RetryJobOutcome.MAX_RETRIES


@pytest.mark.asyncio
async def test_compensate_failed_retry_reschedule_exact_generation(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.QUEUED.value,
        retry_count=2,
        retryable=False,
        source_s3_key="uploads/a.pdf",
    )
    outcome = await store.compensate_failed_retry_reschedule(
        "job-1",
        "user-1",
        retry_count=2,
        retryable=True,
    )
    assert outcome.outcome is store.RetryCompensationOutcome.APPLIED
    assert storage["status"] == PipelineStatus.FAILED.value
    assert storage["retryable"] is True
    assert storage["error_code"] == "pipeline_retry_reschedule_failed"
    assert storage["retry_count"] == 2
    # Filter shape must pin exact generation + QUEUED + owner.
    last = fake_doc.update_calls[-1]
    assert last["query"]["retry_count"] == 2
    assert last["query"]["status"] == PipelineStatus.QUEUED.value
    assert last["query"]["user_id"] == "user-1"
    assert last["query"]["cancel_requested"] == {"$ne": True}


@pytest.mark.asyncio
async def test_compensate_failed_retry_reschedule_cas_miss_safe(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.FAILED.value,
        retry_count=2,
        retryable=True,
        source_s3_key="uploads/a.pdf",
        error_code="other",
    )
    outcome = await store.compensate_failed_retry_reschedule(
        "job-1",
        "user-1",
        retry_count=2,
        retryable=True,
    )
    assert outcome.outcome is store.RetryCompensationOutcome.ALREADY_TERMINAL
    assert storage["error_code"] == "other"
    assert storage["status"] == PipelineStatus.FAILED.value


@pytest.mark.asyncio
async def test_compensate_misses_when_retry_count_differs(fake_doc):
    storage = _seed(
        fake_doc,
        status=PipelineStatus.QUEUED.value,
        retry_count=3,
        retryable=False,
        source_s3_key="uploads/a.pdf",
    )
    outcome = await store.compensate_failed_retry_reschedule(
        "job-1",
        "user-1",
        retry_count=2,
        retryable=True,
    )
    assert outcome.outcome is store.RetryCompensationOutcome.STALE_GENERATION
    assert storage["status"] == PipelineStatus.QUEUED.value
    assert storage["retry_count"] == 3
