import asyncio

from api.core.content_pipeline.application.stages.embedding import (
    compute_concept_embeddings,
    resolve_embedding_settings,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


class _EmbeddingClientStub:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size


def test_resolve_embedding_settings_falls_back_to_legacy_defaults():
    model_name, batch_size = resolve_embedding_settings()

    assert isinstance(model_name, str)
    assert model_name
    assert isinstance(batch_size, int)
    assert batch_size > 0


def test_compute_concept_embeddings_updates_progress_and_uses_factory():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls: list[tuple[PipelineStatus, str, float]] = []
    compute_calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    def compute_embedding_for_concepts(concepts, embed_client):
        compute_calls.append((concepts, embed_client.model_name, embed_client.batch_size))

    concepts = ["c1", "c2"]
    asyncio.run(
        compute_concept_embeddings(
            job,
            concepts=concepts,
            embedding_client_factory=_EmbeddingClientStub,
            compute_embedding_for_concepts=compute_embedding_for_concepts,
            persist_job_state=persist_job_state,
            model_name="model-x",
            batch_size=16,
        )
    )

    assert compute_calls == [(concepts, "model-x", 16)]
    assert calls == [
        (PipelineStatus.EMBEDDING, "Computing embeddings...", 0.35),
        (PipelineStatus.EMBEDDING, "Computing embeddings...", 0.45),
    ]
