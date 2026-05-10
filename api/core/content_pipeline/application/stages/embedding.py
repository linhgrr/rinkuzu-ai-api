"""Embedding stage for the content pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from api.config import get_settings
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineProgress, PipelineStatus

from .execution import resolve_timeout_policy
from .model_worker import encode_texts_with_sentence_transformer_worker

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def resolve_embedding_settings() -> tuple[str, int]:
    """Read embedding settings from the unified backend config."""
    settings = get_settings()
    return settings.embedding_model, settings.embedding_batch_size


async def compute_concept_embeddings(
    job: PipelineJob,
    *,
    concepts: list[Any],
    persist_job_state: PersistJobStateFn,
    model_name: str,
    batch_size: int,
) -> None:
    """Compute embeddings for extracted concepts and persist stage progress."""
    await persist_job_state(
        job, PipelineStatus.EMBEDDING, "Computing embeddings...", PipelineProgress.EMBEDDING_START
    )

    settings = get_settings()
    _, stage_timeout = resolve_timeout_policy()

    concepts_with_names = []
    name_texts = []
    concepts_with_definitions = []
    definition_texts = []

    for concept in concepts:
        concept_name = getattr(concept, "name", "")
        if not concept_name:
            continue
        concepts_with_names.append(concept)
        name_texts.append(str(concept_name))

        concept_definition = getattr(concept, "definition", "")
        if concept_definition:
            concepts_with_definitions.append(concept)
            definition_texts.append(str(concept_definition))

    if name_texts:
        name_embeddings = await encode_texts_with_sentence_transformer_worker(
            texts=name_texts,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=True,
            use_vi_tokenizer=bool(settings.use_vi_tokenizer),
            max_seq_length=settings.max_seq_length,
            show_progress_bar=False,
            stage_name="embedding_name_generation",
            timeout_sec=stage_timeout,
        )
        for concept, embedding in zip(concepts_with_names, name_embeddings, strict=False):
            concept.name_embedding = embedding

    if definition_texts:
        definition_embeddings = await encode_texts_with_sentence_transformer_worker(
            texts=definition_texts,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=True,
            use_vi_tokenizer=bool(settings.use_vi_tokenizer),
            max_seq_length=settings.max_seq_length,
            show_progress_bar=False,
            stage_name="embedding_definition_generation",
            timeout_sec=stage_timeout,
        )
        for concept, embedding in zip(
            concepts_with_definitions,
            definition_embeddings,
            strict=False,
        ):
            concept.definition_embedding = embedding

    await persist_job_state(
        job, PipelineStatus.EMBEDDING, "Computing embeddings...", PipelineProgress.EMBEDDING_DONE
    )
