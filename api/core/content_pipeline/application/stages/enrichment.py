"""Final content enrichment stages for the content pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from loguru import logger

from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus

from .execution import run_blocking_stage, safe_run

PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]


def build_ordered_embedding_texts(
    concepts_data: dict[str, dict[str, Any]],
    concept_map: dict[str, int],
) -> list[str]:
    """Build ordered concept texts used to encode SAINT embeddings."""
    id_to_concept = {index: concept_id for concept_id, index in concept_map.items()}
    ordered_texts = []
    for index in range(len(concept_map)):
        concept_id = id_to_concept.get(index, str(index))
        name = concepts_data[concept_id]["name"]
        definition = concepts_data[concept_id].get("definition", "")
        ordered_texts.append(f"{name}: {definition}" if definition else name)
    return ordered_texts


async def generate_saint_concept_embeddings(
    job: PipelineJob,
    *,
    concepts_data: dict[str, dict[str, Any]],
    concept_map: dict[str, int],
    text_model_factory: Callable[[], Any],
    persist_job_state: PersistJobStateFn,
    ) -> list[list[float]] | None:
    """Generate concept embeddings for downstream adaptive-learning models."""
    await persist_job_state(
        job,
        PipelineStatus.OPTIMIZING,
        "Generating concept embeddings for SAINT...",
        0.92,
    )
    async def _generate():
        text_model = text_model_factory()
        ordered_texts = build_ordered_embedding_texts(concepts_data, concept_map)
        embeddings: Any = await run_blocking_stage(
            text_model.encode,
            ordered_texts,
            show_progress_bar=False,
            batch_size=32,
            stage_name="saint_embedding_generation",
        )
        logger.info("[Pipeline] ✓ Generated embeddings for {} concepts", len(ordered_texts))
        return cast("list[list[float]]", embeddings.tolist())

    return await safe_run(
        _generate,
        fail_message="Could not generate embeddings",
    )


async def generate_concept_theories(
    job: PipelineJob,
    *,
    concepts_data: dict[str, dict[str, Any]],
    generate_theory: Callable[[str, str], Any],
    persist_job_state: PersistJobStateFn,
    concurrency: int = 5,
) -> None:
    """Best-effort theory generation for concepts missing precomputed theory."""
    await persist_job_state(
        job,
        PipelineStatus.OPTIMIZING,
        "Generating concept theories...",
        0.93,
    )
    async def _generate_theories():
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_one(concept_id: str, name: str, definition: str):
            async with semaphore:
                theory: Any = await run_blocking_stage(
                    generate_theory,
                    name,
                    definition,
                    stage_name="concept_theory_generation",
                )
                return concept_id, theory

        tasks = [
            generate_one(concept_id, concept_data["name"], concept_data.get("definition", ""))
            for concept_id, concept_data in concepts_data.items()
            if "theory" not in concept_data
        ]

        if tasks:
            logger.info("[Pipeline] Generating theory for {} concepts...", len(tasks))
            results = await asyncio.gather(*tasks)
            for concept_id, theory in results:
                concepts_data[concept_id]["theory"] = theory
            logger.info("[Pipeline] ✓ Theory generation complete")

    await safe_run(
        _generate_theories,
        fail_message="Failed to pre-generate theory",
    )
