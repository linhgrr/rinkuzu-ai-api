import asyncio

from api.core.content_pipeline.application.stages.enrichment import (
    build_ordered_embedding_texts,
    generate_concept_theories,
    generate_saint_concept_embeddings,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_build_ordered_embedding_texts_uses_concept_map_order():
    concepts_data = {
        "c2": {"name": "Beta", "definition": "beta def"},
        "c1": {"name": "Alpha", "definition": "alpha def"},
    }
    concept_map = {"c1": 0, "c2": 1}

    texts = build_ordered_embedding_texts(concepts_data, concept_map)

    assert texts == ["Alpha: alpha def", "Beta: beta def"]


class _TextModelStub:
    def encode(self, ordered_texts, show_progress_bar=False, batch_size=32):
        return _EmbeddingsStub([[len(text)] for text in ordered_texts])


class _EmbeddingsStub:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def test_generate_saint_concept_embeddings_updates_progress_and_returns_vectors():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    embeddings = asyncio.run(
        generate_saint_concept_embeddings(
            job,
            concepts_data={
                "c1": {"name": "Alpha", "definition": "alpha def"},
                "c2": {"name": "Beta", "definition": ""},
            },
            concept_map={"c1": 0, "c2": 1},
            text_model_factory=_TextModelStub,
            persist_job_state=persist_job_state,
        )
    )

    assert embeddings == [[16], [4]]
    assert calls == [
        (PipelineStatus.OPTIMIZING, "Generating concept embeddings for SAINT...", 0.92),
    ]


def test_generate_concept_theories_fills_missing_theories_only():
    job = PipelineJob(job_id="job-2", filename="lesson.pdf", subject_id="algebra")
    calls = []
    concepts_data = {
        "c1": {"name": "Alpha", "definition": "alpha def"},
        "c2": {"name": "Beta", "definition": "beta def", "theory": {"body": "keep"}},
    }

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def generate_theory(name: str, definition: str):
        return {"body": f"{name}|{definition}"}

    asyncio.run(
        generate_concept_theories(
            job,
            concepts_data=concepts_data,
            generate_theory=generate_theory,
            persist_job_state=persist_job_state,
            concurrency=2,
        )
    )

    assert concepts_data == {
        "c1": {"name": "Alpha", "definition": "alpha def", "theory": {"body": "Alpha|alpha def"}},
        "c2": {"name": "Beta", "definition": "beta def", "theory": {"body": "keep"}},
    }
    assert calls == [
        (PipelineStatus.OPTIMIZING, "Generating concept theories...", 0.93),
    ]
