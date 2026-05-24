"""Tests for the (deprecated) embedding stage.

After the swap to ``MLPPrerequisiteRanker`` the stage is a no-op kept only
to preserve the existing pipeline_runner call site. These tests pin the
no-op contract so future refactors don't accidentally re-enable
vietnamese-sbert encoding.
"""

import asyncio

from api.core.content_pipeline.application.stages.embedding import (
    compute_concept_embeddings,
    resolve_embedding_settings,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_resolve_embedding_settings_returns_dummy_values():
    model_name, batch_size = resolve_embedding_settings()
    assert model_name == ""
    assert batch_size == 0


def test_compute_concept_embeddings_is_a_noop_but_advances_progress():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls: list[tuple[PipelineStatus, str, float]] = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    concepts = [
        type("Concept", (), {"name": "Alpha", "definition": "alpha def"})(),
        type("Concept", (), {"name": "Beta", "definition": ""})(),
    ]
    asyncio.run(
        compute_concept_embeddings(
            job,
            concepts=concepts,
            persist_job_state=persist_job_state,
            model_name="ignored",
            batch_size=0,
        )
    )

    # Stage must NOT mutate concepts.
    for concept in concepts:
        assert not hasattr(concept, "name_embedding") or concept.name_embedding is None
        assert not hasattr(concept, "definition_embedding") or concept.definition_embedding is None

    # Stage still needs to advance pipeline progress so downstream stages run.
    assert len(calls) == 1
    assert calls[0][0] == PipelineStatus.EMBEDDING
