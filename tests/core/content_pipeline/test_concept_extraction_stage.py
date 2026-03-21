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

    def extract_from_batch(self, chunk_texts, subject_id):
        self.calls.append((chunk_texts, subject_id))
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
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    chunks = [
        SimpleNamespace(page_content="chunk one"),
        SimpleNamespace(page_content="chunk two"),
    ]
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
            chunks=chunks,
            extraction_chain=extraction_chain,
            postprocess_concepts=postprocess,
            persist_job_state=persist_job_state,
        )
    )

    assert [concept.concept_id for concept in extracted] == ["c2", "c1"]
    assert extraction_chain.calls == [(["chunk one", "chunk two"], "algebra")]
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
