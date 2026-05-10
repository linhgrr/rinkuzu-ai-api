import asyncio

from pydantic import BaseModel

from api.core.content_pipeline.application.stages import enrichment as enrichment_stage
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

def test_generate_saint_concept_embeddings_updates_progress_and_returns_vectors(monkeypatch):
    job = PipelineJob(job_id="job-1", filename="lesson.pdf", subject_id="algebra")
    calls = []
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

    monkeypatch.setattr(
        enrichment_stage,
        "encode_texts_with_sentence_transformer_worker",
        fake_encode_texts_with_sentence_transformer_worker,
    )

    embeddings = asyncio.run(
        generate_saint_concept_embeddings(
            job,
            concepts_data={
                "c1": {"name": "Alpha", "definition": "alpha def"},
                "c2": {"name": "Beta", "definition": ""},
            },
            concept_map={"c1": 0, "c2": 1},
            persist_job_state=persist_job_state,
        )
    )

    assert embeddings == [[16], [4]]
    assert worker_calls[0][0] == ["Alpha: alpha def", "Beta"]
    assert worker_calls[0][1:4] == ("paraphrase-multilingual-mpnet-base-v2", 32, False)
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


def test_generate_concept_theories_normalizes_pydantic_theory_payloads():
    class _TheoryPayload(BaseModel):
        content: str
        examples: list[str]

    job = PipelineJob(job_id="job-3", filename="lesson.pdf", subject_id="algebra")
    concepts_data = {
        "c1": {"name": "Alpha", "definition": "alpha def"},
    }

    async def persist_job_state(*_args, **_kwargs):
        return None

    def generate_theory(name: str, definition: str):
        return _TheoryPayload(content=f"{name}:{definition}", examples=["Ví dụ A"])

    asyncio.run(
        generate_concept_theories(
            job,
            concepts_data=concepts_data,
            generate_theory=generate_theory,
            persist_job_state=persist_job_state,
            concurrency=1,
        )
    )

    assert concepts_data["c1"]["theory"] == {
        "content": "Alpha:alpha def",
        "examples": ["Ví dụ A"],
    }
