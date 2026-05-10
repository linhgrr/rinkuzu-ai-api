import asyncio

import pytest

from api.core.content_pipeline.application.stages import embedding as embedding_stage
from api.core.content_pipeline.application.stages.embedding import (
    compute_concept_embeddings,
    resolve_embedding_settings,
)
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus


def test_resolve_embedding_settings_reads_unified_backend_config(monkeypatch):
    class _SettingsStub:
        embedding_model = "model-from-settings"
        embedding_batch_size = 48

    monkeypatch.setattr(embedding_stage, "get_settings", _SettingsStub)

    model_name, batch_size = resolve_embedding_settings()

    assert model_name == "model-from-settings"
    assert batch_size == 48


def test_compute_concept_embeddings_updates_progress_and_uses_worker():
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls: list[tuple[PipelineStatus, str, float]] = []
    worker_calls = []

    async def persist_job_state(job_arg, status, step, progress):
        assert job_arg is job
        calls.append((status, step, progress))

    async def fake_encode_texts_with_sentence_transformer_worker(
        *,
        texts,
        model_name,
        batch_size,
        normalize_embeddings,
        use_vi_tokenizer,
        max_seq_length,
        show_progress_bar,
        stage_name,
        timeout_sec,
    ):
        worker_calls.append(
            (
                texts,
                model_name,
                batch_size,
                normalize_embeddings,
                use_vi_tokenizer,
                max_seq_length,
                show_progress_bar,
                stage_name,
                timeout_sec,
            )
        )
        return [[len(text)] for text in texts]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        embedding_stage,
        "encode_texts_with_sentence_transformer_worker",
        fake_encode_texts_with_sentence_transformer_worker,
    )

    try:
        concepts = [
            type("Concept", (), {"name": "Alpha", "definition": "alpha def"})(),
            type("Concept", (), {"name": "Beta", "definition": ""})(),
        ]
        asyncio.run(
            compute_concept_embeddings(
                job,
                concepts=concepts,
                persist_job_state=persist_job_state,
                model_name="model-x",
                batch_size=16,
            )
        )
    finally:
        monkeypatch.undo()

    assert worker_calls[0][0] == ["Alpha", "Beta"]
    assert worker_calls[0][1:4] == ("model-x", 16, True)
    assert concepts[0].name_embedding == [5]
    assert concepts[0].definition_embedding == [9]
    assert concepts[1].name_embedding == [4]
    assert calls == [
        (PipelineStatus.EMBEDDING, "Computing embeddings...", 0.35),
        (PipelineStatus.EMBEDDING, "Computing embeddings...", 0.45),
    ]
