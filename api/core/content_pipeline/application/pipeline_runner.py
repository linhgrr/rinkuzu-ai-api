"""Composition layer for running the unified content pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.core.content_pipeline.infrastructure.runtime import (
    calculate_file_hash,
    get_content_processor_bindings,
    get_s3_client,
)

from .ports import LoadJobFn, PersistJobStateFn, SaveJobFn  # noqa: TC001
from .stages.cache_restore import (
    try_restore_completed_job_from_mongo,
    try_restore_completed_job_from_s3,
)
from .stages.chunk_persistence import persist_document_chunks
from .stages.concept_extraction import _resolve_extraction_timeout, extract_concepts_from_chunks
from .stages.concept_merge import merge_duplicate_concepts
from .stages.document_loading import load_document_chunks
from .stages.embedding import compute_concept_embeddings, resolve_embedding_settings
from .stages.enrichment import (
    generate_concept_theories,
    generate_saint_concept_embeddings,
)
from .stages.execution import resolve_timeout_policy
from .stages.finalization import (
    complete_pipeline_job,
    persist_terminal_failure,
    upload_result_cache,
)
from .stages.graph_building import build_knowledge_graph
from .stages.graph_optimization import optimize_graph
from .stages.result_assembly import (
    assemble_pipeline_result,
    serialize_concepts,
    serialize_prerequisite_edges,
)


def populate_job_metrics_from_result(job: PipelineJob) -> None:
    """Populate top-level job metrics from a completed result payload."""
    if not isinstance(job.result, dict):
        return

    concept_map = job.result.get("concept_map")
    prereq_edges = job.result.get("prereq_edges")
    stats = job.result.get("stats")

    if not isinstance(concept_map, dict):
        concept_map = {}
    if not isinstance(prereq_edges, list):
        prereq_edges = []
    if not isinstance(stats, dict):
        stats = {}

    num_nodes = int(stats.get("num_nodes", len(concept_map)))
    num_edges = int(stats.get("num_edges", len(prereq_edges)))

    if job.concepts_extracted <= 0:
        job.concepts_extracted = len(concept_map)
    job.concepts_after_merge = max(job.concepts_after_merge, len(concept_map), num_nodes)
    if job.relations_verified <= 0:
        job.relations_verified = int(stats.get("relations_verified", len(prereq_edges)))

    merged_stats = dict(stats)
    merged_stats["num_nodes"] = num_nodes
    merged_stats["num_edges"] = num_edges
    merged_stats["is_dag"] = bool(stats.get("is_dag", True))
    job.graph_stats = merged_stats


async def _resolve_effective_job_timeout(
    *,
    file_path: str,
    job: PipelineJob,
    settings: Any,
) -> float | None:
    job_timeout_sec, stage_timeout_sec = resolve_timeout_policy()
    if job_timeout_sec is None:
        return None

    extraction_timeout_sec = await _resolve_extraction_timeout(file_path, job, settings)
    stage_buffer_sec = max(float(stage_timeout_sec or 0.0), 300.0)
    effective_timeout_sec = max(job_timeout_sec, extraction_timeout_sec + (stage_buffer_sec * 2))
    logger.info(
        "[PipelineRunner] effective_job_timeout_sec={} configured_job_timeout_sec={} extraction_timeout_sec={} stage_buffer_sec={}",
        effective_timeout_sec,
        job_timeout_sec,
        extraction_timeout_sec,
        stage_buffer_sec,
    )
    return effective_timeout_sec


class PipelineRunner:
    """Runs the content pipeline by composing extracted stages."""

    def __init__(
        self,
        *,
        load_job: LoadJobFn,
        save_job: SaveJobFn,
        persist_job_state: PersistJobStateFn,
        chunk_chroma_store: Any = None,
    ) -> None:
        self._load_job = load_job
        self._save_job = save_job
        self._persist_job_state = persist_job_state
        self._chunk_chroma_store = chunk_chroma_store

    @staticmethod
    def _cleanup_upload(file_path: str) -> None:
        path = Path(file_path)
        try:
            if path.exists():
                path.unlink()
                logger.debug("[PipelineRunner] Deleted upload {}", file_path)
        except OSError as exc:
            logger.warning("[PipelineRunner] Failed to delete upload {}: {}", file_path, exc)

    async def run(
        self,
        job: PipelineJob,
        *,
        file_path: str,
        prs_threshold: float,
        min_confidence: float,
        apply_reduction: bool,
        page_batch_size: int,
    ) -> None:
        settings = get_settings()
        job.page_batch_size = page_batch_size
        job_timeout_sec = await _resolve_effective_job_timeout(
            file_path=file_path,
            job=job,
            settings=settings,
        )
        try:
            async with asyncio.timeout(job_timeout_sec):
                bindings = get_content_processor_bindings()

                if await try_restore_completed_job_from_mongo(
                    job,
                    load_job=self._load_job,
                    populate_metrics=populate_job_metrics_from_result,
                ):
                    return

                s3_client = get_s3_client()
                bucket_name = settings.object_storage_bucket
                cache_key = await try_restore_completed_job_from_s3(
                    job,
                    file_path=file_path,
                    s3_client=s3_client,
                    bucket_name=bucket_name,
                    hash_file=calculate_file_hash,
                    save_job=self._save_job,
                    populate_metrics=populate_job_metrics_from_result,
                )
                if job.status == PipelineStatus.COMPLETED:
                    try:
                        chunks = await load_document_chunks(
                            job,
                            file_path=file_path,
                            persist_job_state=self._persist_job_state,
                        )
                        await persist_document_chunks(
                            job,
                            chunks=chunks,
                            chunk_chroma_store=self._chunk_chroma_store,
                            persist_job_state=self._persist_job_state,
                        )
                        await complete_pipeline_job(
                            job,
                            persist_job_state=self._persist_job_state,
                        )
                    except Exception:
                        logger.exception(
                            "[Pipeline] Failed to rebuild reusable chunks for cached job {}",
                            job.job_id,
                        )
                    return

                document_text_holder: dict[str, Any] = {}
                chunks = await load_document_chunks(
                    job,
                    file_path=file_path,
                    persist_job_state=self._persist_job_state,
                    document_text_out=document_text_holder,
                )

                # Persist document chunks for RAG (MongoDB + ChromaDB)
                await persist_document_chunks(
                    job,
                    chunks=chunks,
                    chunk_chroma_store=self._chunk_chroma_store,
                    persist_job_state=self._persist_job_state,
                )

                extraction_chain = bindings.extraction_chain_cls()
                relation_engine = bindings.relation_engine_factory(
                    extraction_chain=extraction_chain,
                )
                all_concepts = await extract_concepts_from_chunks(
                    job,
                    file_path=file_path,
                    extraction_chain=extraction_chain,
                    postprocess_concepts=bindings.postprocess_concepts,
                    persist_job_state=self._persist_job_state,
                    document_text=document_text_holder.get("document_text"),
                )
                failure_ratio = (
                    job.failed_batch_count / job.batch_count if job.batch_count > 0 else 0.0
                )
                if job.batch_count > 0 and (
                    failure_ratio > settings.content_pipeline_batch_failure_ratio_threshold
                ):
                    raise RuntimeError(
                        "Too many PDF concept-extraction batches failed "
                        f"({job.failed_batch_count}/{job.batch_count})."
                    )
                if not all_concepts:
                    raise RuntimeError("No concepts were extracted from the document.")

                model_name, batch_size = resolve_embedding_settings()
                await compute_concept_embeddings(
                    job,
                    concepts=all_concepts,
                    persist_job_state=self._persist_job_state,
                    model_name=model_name,
                    batch_size=batch_size,
                )

                all_concepts = await merge_duplicate_concepts(
                    job,
                    concepts=all_concepts,
                    merge_by_name=bindings.merge_by_name,
                    persist_job_state=self._persist_job_state,
                )

                relation_result = await relation_engine.discover_relations(
                    job=job,
                    concepts=all_concepts,
                    prs_threshold=prs_threshold,
                    min_confidence=min_confidence,
                    persist_job_state=self._persist_job_state,
                )

                graph, graph_build_stats = await build_knowledge_graph(
                    job,
                    concepts=all_concepts,
                    verified_relations=relation_result.verified_relations,
                    knowledge_graph_builder_factory=bindings.knowledge_graph_builder_factory,
                    persist_job_state=self._persist_job_state,
                )
                extracted_relation_count = graph_build_stats["extracted_relation_count"]
                verified_relation_count = graph_build_stats["verified_relation_count"]

                graph, optimization_stats = await optimize_graph(
                    job,
                    graph=graph,
                    concepts=all_concepts,
                    apply_reduction=apply_reduction,
                    make_dag_with_llm=bindings.make_dag_with_llm,
                    apply_transitive_reduction=bindings.apply_transitive_reduction,
                    persist_job_state=self._persist_job_state,
                )

                stats = dict(graph_build_stats.get("base_graph_stats") or {})
                stats["num_nodes"] = optimization_stats["num_nodes"]
                stats["num_edges"] = optimization_stats["num_edges"]
                stats["is_dag"] = optimization_stats["is_dag"]
                stats["relations_from_extraction"] = extracted_relation_count
                stats["relations_from_verification"] = verified_relation_count
                stats["relations_verified"] = job.relations_verified
                if optimization_stats.get("cycle_stats") is not None:
                    stats["cycle_stats"] = optimization_stats["cycle_stats"]
                job.graph_stats = stats

                concepts_data, concept_map = serialize_concepts(all_concepts)
                prereq_edges = serialize_prerequisite_edges(graph, concept_map)

                concept_embeddings_list = await generate_saint_concept_embeddings(
                    job,
                    concepts_data=concepts_data,
                    concept_map=concept_map,
                    persist_job_state=self._persist_job_state,
                )

                await generate_concept_theories(
                    job,
                    concepts_data=concepts_data,
                    generate_theory=bindings.generate_theory,
                    persist_job_state=self._persist_job_state,
                )

                job.result = assemble_pipeline_result(
                    concepts_data=concepts_data,
                    concept_map=concept_map,
                    prereq_edges=prereq_edges,
                    concept_embeddings=concept_embeddings_list,
                    stats=stats,
                )
                job.result["page_batch_size"] = job.page_batch_size
                job.result["batch_count"] = job.batch_count
                job.result["failed_batches"] = list(
                    getattr(extraction_chain, "last_failed_batches", [])
                )
                job.result["warnings"] = [
                    item["reason"]
                    for item in getattr(extraction_chain, "last_failed_batches", [])
                    if item.get("reason")
                ]
                job.result["partial_success"] = job.partial_success
                populate_job_metrics_from_result(job)

                await complete_pipeline_job(
                    job,
                    persist_job_state=self._persist_job_state,
                )
                await upload_result_cache(
                    result=job.result,
                    s3_client=s3_client,
                    bucket_name=bucket_name,
                    cache_key=cache_key,
                )
        except asyncio.CancelledError as exc:
            await persist_terminal_failure(
                job,
                error=exc,
                save_job=self._save_job,
            )
            return
        except BaseException as exc:
            await persist_terminal_failure(
                job,
                error=exc,
                save_job=self._save_job,
            )
        finally:
            self._cleanup_upload(file_path)
