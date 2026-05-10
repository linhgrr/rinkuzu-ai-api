import asyncio
from types import SimpleNamespace

from api.core.content_pipeline.application.stages.concept_extraction import (
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

    async def extract_from_document(self, file_path, subject_id, page_batch_size, *, job_id=None):
        self.calls.append((file_path, subject_id, page_batch_size, job_id))
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
            file_path="/tmp/lesson.pdf",  # noqa: S108
            extraction_chain=extraction_chain,
            postprocess_concepts=postprocess,
            persist_job_state=persist_job_state,
        )
    )

    assert [concept.concept_id for concept in extracted] == ["c2", "c1"]
    assert extraction_chain.calls == [("/tmp/lesson.pdf", "algebra", 10, "job-1")]  # noqa: S108
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
