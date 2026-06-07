import asyncio
from types import SimpleNamespace

from api.core.content_pipeline.application.stages.concept_extraction import (
    _resolve_extraction_timeout,
    build_partial_concept_graph,
    extract_concepts_from_chunks,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


class _FakeExtractionChain:
    def __init__(self, response):
        self._response = response
        self.calls = []
        self.last_batches = [{"batch_index": 0}, {"batch_index": 1}, {"batch_index": 2}]
        self.last_failed_batches = [{"batch_index": 2, "reason": "Error: boom"}]

    async def extract_from_document(
        self,
        file_path,
        subject_id,
        page_batch_size,
        *,
        document_text=None,
        job_id=None,
        on_batch_progress=None,
    ):
        del on_batch_progress
        self.calls.append((file_path, subject_id, page_batch_size, document_text, job_id))
        return self._response


def test_build_partial_concept_graph_serializes_basic_node_data():
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]

    graph = build_partial_concept_graph(concepts)

    assert graph == {
        "nodes": [
            {"id": "c1", "name": "Alpha"},
            {"id": "c2", "name": "Beta"},
        ],
        "edges": [],
    }


def test_extract_concepts_from_chunks_updates_job_metrics_and_progress():
    job = PipelineJob(
        job_id="job-1",
        filename="lesson.pdf",
        subject_id="algebra",
        page_batch_size=10,
    )
    concepts = [
        SimpleNamespace(concept_id="c1", name="Alpha"),
        SimpleNamespace(concept_id="c2", name="Beta"),
    ]
    extraction_chain = _FakeExtractionChain(
        [
            SimpleNamespace(concepts=[concepts[0]]),
            SimpleNamespace(concepts=[concepts[1]]),
            None,
        ]
    )
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def postprocess(items):
        return list(reversed(items))

    extracted = asyncio.run(
        extract_concepts_from_chunks(
            job,
            file_path="/tmp/lesson.pdf",
            extraction_chain=extraction_chain,
            postprocess_concepts=postprocess,
            persist_job_state=persist_job_state,
        )
    )

    assert [concept.concept_id for concept in extracted.concepts] == ["c2", "c1"]
    assert extracted.failed_batches == [{"batch_index": 2, "reason": "Error: boom"}]
    assert extracted.warnings == ["Error: boom"]
    assert extraction_chain.calls == [("/tmp/lesson.pdf", "algebra", 10, None, "job-1")]
    assert job.batch_count == 3
    assert job.failed_batch_count == 1
    assert job.partial_success is True
    assert job.concepts_extracted == 2
    assert job.partial_graph == {
        "nodes": [
            {"id": "c2", "name": "Beta"},
            {"id": "c1", "name": "Alpha"},
        ],
        "edges": [],
    }
    assert calls == [
        (PipelineStatus.EXTRACTING, "Extracting concepts with LLM...", 0.15),
        (PipelineStatus.EXTRACTING, "Extracting concepts with LLM...", 0.30),
    ]


class _BatchHeartbeatExtractionChain:
    """Fake chain that drives ``on_batch_progress`` once per simulated batch."""

    def __init__(self, response, total_batches):
        self._response = response
        self._total_batches = total_batches
        self.last_batches = [{"batch_index": i} for i in range(total_batches)]
        self.last_failed_batches = []

    async def extract_from_document(
        self,
        file_path,
        subject_id,
        page_batch_size,
        *,
        document_text=None,
        job_id=None,
        on_batch_progress=None,
    ):
        del file_path, subject_id, page_batch_size, document_text, job_id
        for done in range(1, self._total_batches + 1):
            if on_batch_progress is not None:
                await on_batch_progress(done, self._total_batches)
        return self._response


def test_extract_concepts_emits_per_batch_heartbeat():
    job = PipelineJob(
        job_id="job-hb",
        filename="lesson.pdf",
        subject_id="algebra",
        page_batch_size=10,
    )
    total_batches = 3
    extraction_chain = _BatchHeartbeatExtractionChain(
        [SimpleNamespace(concepts=[], notes=None) for _ in range(total_batches)],
        total_batches=total_batches,
    )
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    asyncio.run(
        extract_concepts_from_chunks(
            job,
            file_path="/tmp/lesson.pdf",
            extraction_chain=extraction_chain,
            postprocess_concepts=lambda items: items,
            persist_job_state=persist_job_state,
        )
    )

    # START + 3 per-batch heartbeats + DONE = 5 persist calls.
    assert len(calls) == total_batches + 2
    progresses = [progress for _, _, progress in calls]
    assert progresses[0] == 0.15
    assert progresses[-1] == 0.30
    # Mid-extraction heartbeats interpolate strictly within [0.15, 0.30].
    heartbeat_progresses = progresses[1:-1]
    assert len(heartbeat_progresses) == total_batches
    for value in heartbeat_progresses:
        assert 0.15 <= value <= 0.30
    # Monotonically increasing across the whole sequence.
    assert progresses == sorted(progresses)
    # Per-batch steps carry the batch counter so liveness can be confirmed.
    steps = [step for _, step, _ in calls]
    assert steps[1] == "Extracting concepts with LLM... (1/3 batches)"
    assert steps[3] == "Extracting concepts with LLM... (3/3 batches)"


def test_resolve_extraction_timeout_uses_retry_aware_llm_budget(monkeypatch):
    job = PipelineJob(
        job_id="job-timeout",
        filename="lesson.pdf",
        subject_id="algebra",
        page_batch_size=10,
    )

    async def fake_run_process_stage(*_args, **_kwargs):
        return 25

    monkeypatch.setattr(
        "api.core.content_pipeline.application.stages.concept_extraction.run_process_stage",
        fake_run_process_stage,
    )
    monkeypatch.setattr(
        "api.core.content_pipeline.application.stages.concept_extraction.resolve_timeout_policy",
        lambda: (1800.0, 300.0),
    )

    timeout = asyncio.run(
        _resolve_extraction_timeout(
            "/tmp/lesson.pdf",
            job,
            SimpleNamespace(
                content_pipeline_extraction_secs_per_page=20.0,
                content_pipeline_pdf_page_batch_size=10,
                content_pipeline_llm_request_timeout_sec=60.0,
                content_pipeline_llm_retry_attempts=3,
                content_pipeline_llm_retry_backoff_sec=2.0,
            ),
        )
    )

    assert timeout == 738.0
    assert job.total_pages == 25
