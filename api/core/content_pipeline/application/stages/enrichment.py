"""Final content enrichment stages for the content pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from loguru import logger

from ...domain.jobs import PipelineJob, PipelineStatus
from .execution import run_blocking_stage


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
    try:
        text_model = text_model_factory()
        ordered_texts = build_ordered_embedding_texts(concepts_data, concept_map)
        embeddings = await run_blocking_stage(
            text_model.encode,
            ordered_texts,
            show_progress_bar=False,
            batch_size=32,
            stage_name="saint_embedding_generation",
        )
        logger.info(f"[Pipeline] ✓ Generated embeddings for {len(ordered_texts)} concepts")
        return embeddings.tolist()
    except Exception as exc:
        logger.warning(f"[Pipeline] ⚠ Could not generate embeddings: {exc}")
        return None


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
    try:
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_one(concept_id: str, name: str, definition: str):
            async with semaphore:
                theory = await run_blocking_stage(
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
            logger.info(f"[Pipeline] Generating theory for {len(tasks)} concepts...")
            results = await asyncio.gather(*tasks)
            for concept_id, theory in results:
                concepts_data[concept_id]["theory"] = theory
            logger.info("[Pipeline] ✓ Theory generation complete")
    except Exception as exc:
        logger.warning(f"[Pipeline] ⚠ Failed to pre-generate theory: {exc}")
