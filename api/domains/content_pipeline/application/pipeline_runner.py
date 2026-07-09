"""Composition layer for running the unified content pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from api.config import get_settings
from api.domains.content_pipeline.domain.jobs import PipelineJob, PipelineStatus
from api.domains.content_pipeline.infrastructure.runtime import (
    calculate_file_hash,
    get_content_processor_bindings,
    get_s3_client,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from api.domains.content_pipeline.domain.relations import (
        JsonObject,
        PipelineDebugArtifact,
        PipelineDebugTraceEntry,
    )

from .cancellation import JobCancelledError, raise_if_cancelled
from .ports import LoadCancelFlagFn, LoadJobFn, PersistJobStateFn, SaveJobFn  # noqa: TC001
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
from .stages.quality_gate import validate_pipeline_quality
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


def _concept_debug_sample(concepts: list[Any], *, limit: int = 8) -> list[dict[str, Any]]:
    sample = []
    for concept in concepts[:limit]:
        definition = str(getattr(concept, "definition", "") or "")
        sample.append(
            {
                "concept_id": getattr(concept, "concept_id", None),
                "name": getattr(concept, "name", None),
                "definition_length": len(definition),
                "relation_count": len(getattr(concept, "relations", []) or []),
            }
        )
    return sample


def _graph_debug_summary(graph: Any) -> dict[str, Any]:
    return {
        "num_nodes": graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else None,
        "num_edges": graph.number_of_edges() if hasattr(graph, "number_of_edges") else None,
        "edge_sample": [
            {"source": source_id, "target": target_id, **dict(data)}
            for source_id, target_id, data in list(graph.edges(data=True))[:12]
        ]
        if hasattr(graph, "edges")
        else [],
    }


def _bounded_payload(payload: dict[str, Any]) -> JsonObject:
    """Keep debug payloads JSON-serializable and bounded for Mongo/status polling."""
    return {
        key: value
        for key, value in payload.items()
        if value is None or isinstance(value, (str, int, float, bool, list, dict))
    }


def _truncate_debug_text(value: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _normalize_debug_artifact(
    artifact: PipelineDebugArtifact,
    *,
    max_chars: int,
) -> PipelineDebugArtifact:
    content, truncated = _truncate_debug_text(
        str(artifact.get("content") or ""), max_chars=max_chars
    )
    normalized: PipelineDebugArtifact = {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "kind": str(artifact.get("kind") or ""),
        "label": str(artifact.get("label") or ""),
        "index": int(artifact.get("index") or 0),
        "input": _bounded_payload(dict(artifact.get("input") or {})),
        "output": _bounded_payload(dict(artifact.get("output") or {})),
        "content_type": str(artifact.get("content_type") or "text/plain"),
        "content": content,
        "truncated": bool(artifact.get("truncated", False)) or truncated,
    }
    if "page_start" in artifact:
        normalized["page_start"] = int(artifact["page_start"])
    if "page_end" in artifact:
        normalized["page_end"] = int(artifact["page_end"])
    return normalized


def _ocr_page_debug_artifacts(document_text: Any) -> list[PipelineDebugArtifact]:
    pages = list(getattr(document_text, "pages", []) or [])
    metadata = dict(getattr(document_text, "metadata", {}) or {})
    cache_hit = bool(metadata.get("ocr_cache_hit", False))
    artifacts: list[PipelineDebugArtifact] = []
    for index, page in enumerate(pages):
        page_number = int(getattr(page, "page_number", index + 1) or index + 1)
        text = str(getattr(page, "text", "") or "")
        artifacts.append(
            {
                "artifact_id": f"ocr-page-{index + 1}-page-{page_number}",
                "kind": "ocr_page",
                "label": f"OCR page {page_number}",
                "index": index,
                "page_start": page_number,
                "page_end": page_number,
                "input": {
                    "page_number": page_number,
                    "provider": str(metadata.get("provider") or ""),
                    "model": str(metadata.get("model") or ""),
                    "cache_hit": cache_hit,
                },
                "output": {
                    "char_count": len(text),
                    "empty": not bool(text.strip()),
                },
                "content_type": "text/markdown",
                "content": text,
                "truncated": False,
            }
        )
    return artifacts


class PipelineRunner:
    """Runs the content pipeline by composing extracted stages."""

    def __init__(
        self,
        *,
        load_job: LoadJobFn,
        load_cancel_flag: LoadCancelFlagFn,
        save_job: SaveJobFn,
        persist_job_state: PersistJobStateFn,
        chunk_chroma_store: Any = None,
    ) -> None:
        self._load_job = load_job
        self._load_cancel_flag = load_cancel_flag
        self._save_job = save_job
        self._persist_job_state = persist_job_state
        self._chunk_chroma_store = chunk_chroma_store

    async def _check_cancelled(self, job: PipelineJob) -> None:
        """Re-read the job document from Mongo and raise if cancel was requested.

        The in-memory ``job`` object is not updated by a concurrent API write,
        so we must fetch the latest cancel flag here before checking it. The
        projection read fetches only ``cancel_requested`` instead of the full
        document (graph + embeddings + partial_graph), keeping this hot-path
        checkpoint cheap.
        """
        if await self._load_cancel_flag(job.job_id):
            job.cancel_requested = True
        raise_if_cancelled(job)

    async def _trace_step(
        self,
        job: PipelineJob,
        *,
        step_id: str,
        label: str,
        input_payload: dict[str, Any],
        run: Callable[[], Awaitable[Any]],
        output_payload: Callable[[Any], dict[str, Any]],
        artifacts_payload: Callable[[Any], list[PipelineDebugArtifact]] | None = None,
    ) -> Any:
        """Decorator-style stage instrumentation kept out of stage implementations."""
        started_at = time.time()
        entry = cast(
            "PipelineDebugTraceEntry",
            {
                "step_id": step_id,
                "label": label,
                "status": "running",
                "started_at": started_at,
                "completed_at": None,
                "duration_ms": None,
                "input": _bounded_payload(input_payload),
                "output": None,
                "error": None,
                "artifacts": [],
            },
        )
        job.debug_trace = [item for item in job.debug_trace if item.get("step_id") != step_id]
        job.debug_trace.append(entry)
        await self._save_job(job)

        try:
            result = await run()
        except BaseException as exc:
            completed_at = time.time()
            entry["status"] = "failed"
            entry["completed_at"] = completed_at
            entry["duration_ms"] = round((completed_at - started_at) * 1000, 2)
            entry["error"] = str(exc)
            await self._save_job(job)
            raise

        completed_at = time.time()
        entry["status"] = "completed"
        entry["completed_at"] = completed_at
        entry["duration_ms"] = round((completed_at - started_at) * 1000, 2)
        entry["output"] = _bounded_payload(output_payload(result))
        if artifacts_payload is not None:
            entry["artifacts"] = [
                _normalize_debug_artifact(
                    artifact,
                    max_chars=get_settings().content_pipeline_debug_artifact_max_chars,
                )
                for artifact in artifacts_payload(result)
            ]
        await self._save_job(job)
        return result

    async def _record_debug_artifact(
        self,
        job: PipelineJob,
        *,
        step_id: str,
        artifact: PipelineDebugArtifact,
    ) -> None:
        """Attach one detailed audit artifact to a running trace entry."""
        normalized = _normalize_debug_artifact(
            artifact,
            max_chars=get_settings().content_pipeline_debug_artifact_max_chars,
        )
        for entry in job.debug_trace:
            if entry.get("step_id") == step_id:
                artifacts = entry.setdefault("artifacts", [])
                artifacts = artifacts if isinstance(artifacts, list) else []
                artifact_id = normalized.get("artifact_id")
                entry["artifacts"] = [
                    item for item in artifacts if item.get("artifact_id") != artifact_id
                ]
                entry["artifacts"].append(normalized)
                await self._save_job(job)
                return

    async def _persist_cancelled(self, job: PipelineJob, exc: JobCancelledError) -> None:
        """Persist CANCELLED terminal state — distinct from FAILED."""
        import time as _time

        now = _time.time()
        job.error_code = "pipeline_cancelled"
        job.user_message = "Processing was cancelled."
        job.retryable = True
        job.mark_cancelled(str(exc))
        job.completed_at = now
        job.updated_at = now
        job.heartbeat_at = now
        await self._save_job(job)

    @staticmethod
    def _cleanup_upload(file_path: str) -> None:
        path = Path(file_path)
        try:
            if path.exists():
                path.unlink()
                logger.debug("[PipelineRunner] Deleted upload {}", file_path)
        except OSError as exc:
            logger.warning("[PipelineRunner] Failed to delete upload {}: {}", file_path, exc)

    async def _rebuild_reusable_chunks(self, job: PipelineJob, *, file_path: str) -> None:
        """Reload + persist document chunks for a cache-restored job (RAG reuse)."""
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
        await complete_pipeline_job(job, persist_job_state=self._persist_job_state)

    async def run(
        self,
        job: PipelineJob,
        *,
        file_path: str,
        prs_threshold: float | None,
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
                        await self._rebuild_reusable_chunks(job, file_path=file_path)
                    except Exception:
                        logger.exception(
                            "[Pipeline] Failed to rebuild reusable chunks for cached job {}",
                            job.job_id,
                        )
                    return

                document_text_holder: dict[str, Any] = {}
                chunks = await self._trace_step(
                    job,
                    step_id="document_loading",
                    label="Load PDF and build chunks",
                    input_payload={
                        "filename": job.filename,
                        "file_path": str(Path(file_path).name),
                        "page_batch_size": page_batch_size,
                    },
                    run=lambda: load_document_chunks(
                        job,
                        file_path=file_path,
                        persist_job_state=self._persist_job_state,
                        document_text_out=document_text_holder,
                    ),
                    output_payload=lambda result: {
                        "chunk_count": len(result),
                        "total_chunks": job.total_chunks,
                        "total_pages": job.total_pages,
                        "document_text_available": bool(document_text_holder.get("document_text")),
                    },
                    artifacts_payload=lambda _result: _ocr_page_debug_artifacts(
                        document_text_holder.get("document_text")
                    ),
                )

                # Checkpoint 1: after document loading
                await self._check_cancelled(job)

                # Persist document chunks for RAG (MongoDB + ChromaDB)
                await self._trace_step(
                    job,
                    step_id="chunk_persistence",
                    label="Persist chunks for retrieval",
                    input_payload={
                        "chunk_count": len(chunks),
                        "chunk_store_enabled": self._chunk_chroma_store is not None,
                    },
                    run=lambda: persist_document_chunks(
                        job,
                        chunks=chunks,
                        chunk_chroma_store=self._chunk_chroma_store,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda result: {
                        "persisted": True,
                        "result": result,
                    },
                )

                extraction_chain = bindings.extraction_chain_cls()
                relation_engine = bindings.relation_engine_factory(
                    extraction_chain=extraction_chain,
                )
                outcome = await self._trace_step(
                    job,
                    step_id="concept_extraction",
                    label="Extract concepts and raw prerequisite candidates",
                    input_payload={
                        "chunk_count": len(chunks),
                        "page_batch_size": page_batch_size,
                        "document_text_reused": bool(document_text_holder.get("document_text")),
                    },
                    run=lambda: extract_concepts_from_chunks(
                        job,
                        file_path=file_path,
                        extraction_chain=extraction_chain,
                        postprocess_concepts=bindings.postprocess_concepts,
                        persist_job_state=self._persist_job_state,
                        document_text=document_text_holder.get("document_text"),
                        record_debug_artifact=lambda artifact: self._record_debug_artifact(
                            job,
                            step_id="concept_extraction",
                            artifact=artifact,
                        ),
                    ),
                    output_payload=lambda result: {
                        "concept_count": len(result.concepts),
                        "concept_sample": _concept_debug_sample(result.concepts),
                        "failed_batches": result.failed_batches,
                        "warnings": result.warnings,
                    },
                )
                all_concepts = outcome.concepts

                # Checkpoint 2: after concept extraction
                await self._check_cancelled(job)

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
                await self._trace_step(
                    job,
                    step_id="embedding",
                    label="Compute concept embeddings",
                    input_payload={
                        "concept_count": len(all_concepts),
                        "model_name": model_name,
                        "batch_size": batch_size,
                    },
                    run=lambda: compute_concept_embeddings(
                        job,
                        concepts=all_concepts,
                        persist_job_state=self._persist_job_state,
                        model_name=model_name,
                        batch_size=batch_size,
                    ),
                    output_payload=lambda _result: {
                        "concept_count": len(all_concepts),
                        "concept_sample": _concept_debug_sample(all_concepts),
                    },
                )

                # Checkpoint 3: after embeddings
                await self._check_cancelled(job)

                all_concepts = await self._trace_step(
                    job,
                    step_id="concept_merge",
                    label="Merge duplicate concepts",
                    input_payload={
                        "concept_count_before": len(all_concepts),
                        "concept_sample_before": _concept_debug_sample(all_concepts),
                    },
                    run=lambda: merge_duplicate_concepts(
                        job,
                        concepts=all_concepts,
                        merge_by_name=bindings.merge_by_name,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda result: {
                        "concept_count_after": len(result),
                        "concept_sample_after": _concept_debug_sample(result),
                    },
                )

                # Checkpoint 4: after concept merge
                await self._check_cancelled(job)

                relation_result = await self._trace_step(
                    job,
                    step_id="relation_discovery",
                    label="Rank and verify prerequisite candidates",
                    input_payload={
                        "concept_count": len(all_concepts),
                        "prs_threshold": prs_threshold,
                        "min_confidence": min_confidence,
                        "candidate_sources": ["mlp", "extraction"],
                    },
                    run=lambda: relation_engine.discover_relations(
                        job=job,
                        concepts=all_concepts,
                        prs_threshold=prs_threshold,
                        min_confidence=min_confidence,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda result: {
                        "candidate_count": len(result.candidates),
                        "candidate_sample": [
                            {
                                "source_id": candidate.source_id,
                                "target_id": candidate.target_id,
                                "sources": sorted(candidate.sources),
                                "ranker_score": candidate.ranker_score,
                                "extraction_confidence": candidate.extraction_confidence,
                            }
                            for candidate in result.candidates[:12]
                        ],
                        "verified_count": len(result.verified_relations),
                        "verified_sample": [
                            {
                                "source_id": relation.source_id,
                                "target_id": relation.target_id,
                                "confidence": relation.confidence,
                                "sources": sorted(relation.sources),
                            }
                            for relation in result.verified_relations[:12]
                        ],
                    },
                )

                # Checkpoint 5: after relation discovery
                await self._check_cancelled(job)

                graph, graph_build_stats = await self._trace_step(
                    job,
                    step_id="graph_building",
                    label="Build graph from verified relations",
                    input_payload={
                        "concept_count": len(all_concepts),
                        "verified_relation_count": len(relation_result.verified_relations),
                    },
                    run=lambda: build_knowledge_graph(
                        job,
                        concepts=all_concepts,
                        verified_relations=relation_result.verified_relations,
                        knowledge_graph_builder_factory=bindings.knowledge_graph_builder_factory,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda result: {
                        "graph": _graph_debug_summary(result[0]),
                        "graph_build_stats": result[1],
                    },
                )
                verified_relation_count = graph_build_stats["verified_relation_count"]

                # Checkpoint 6: after graph build
                await self._check_cancelled(job)

                graph, optimization_stats = await self._trace_step(
                    job,
                    step_id="graph_optimization",
                    label="Remove cycles and optimize graph",
                    input_payload={
                        "apply_reduction": apply_reduction,
                        "graph_before": _graph_debug_summary(graph),
                    },
                    run=lambda: optimize_graph(
                        job,
                        graph=graph,
                        concepts=all_concepts,
                        apply_reduction=apply_reduction,
                        make_dag_with_llm=bindings.make_dag_with_llm,
                        apply_transitive_reduction=bindings.apply_transitive_reduction,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda result: {
                        "graph_after": _graph_debug_summary(result[0]),
                        "optimization_stats": result[1],
                    },
                )

                stats = dict(graph_build_stats.get("base_graph_stats") or {})
                stats["num_nodes"] = optimization_stats["num_nodes"]
                stats["num_edges"] = optimization_stats["num_edges"]
                stats["is_dag"] = optimization_stats["is_dag"]
                stats["relation_candidates"] = len(relation_result.candidates)
                stats["relation_candidates_from_extraction"] = sum(
                    1
                    for candidate in relation_result.candidates
                    if "extraction" in candidate.sources
                )
                stats["relation_candidates_from_mlp"] = sum(
                    1 for candidate in relation_result.candidates if "mlp" in candidate.sources
                )
                stats["relations_inserted_after_verification"] = verified_relation_count
                stats["relations_verified"] = job.relations_verified
                if job.graph_stats.get("relations_extraction_candidates_dropped") is not None:
                    stats["relations_extraction_candidates_dropped"] = job.graph_stats[
                        "relations_extraction_candidates_dropped"
                    ]
                if optimization_stats.get("cycle_stats") is not None:
                    stats["cycle_stats"] = optimization_stats["cycle_stats"]

                concepts_data, concept_map = serialize_concepts(all_concepts)
                prereq_edges = serialize_prerequisite_edges(graph, concept_map)
                quality_report = await self._trace_step(
                    job,
                    step_id="quality_gate",
                    label="Validate KG quality gate",
                    input_payload={
                        "concept_count": len(all_concepts),
                        "candidate_count": len(relation_result.candidates),
                        "verified_relation_count": len(relation_result.verified_relations),
                        "extraction_failure_ratio": failure_ratio,
                        "graph": _graph_debug_summary(graph),
                    },
                    run=lambda: asyncio.sleep(
                        0,
                        result=validate_pipeline_quality(
                            job,
                            graph=graph,
                            concepts=all_concepts,
                            concept_ids=set(concept_map),
                            extraction_failure_ratio=failure_ratio,
                            max_extraction_failure_ratio=(
                                settings.content_pipeline_batch_failure_ratio_threshold
                            ),
                            candidate_count=len(relation_result.candidates),
                            verified_relation_count=len(relation_result.verified_relations),
                        ),
                    ),
                    output_payload=lambda result: {"quality_report": result},
                )
                stats["quality_report"] = quality_report
                job.graph_stats = stats

                concept_embeddings_list = await self._trace_step(
                    job,
                    step_id="saint_embeddings",
                    label="Generate SAINT concept embeddings",
                    input_payload={
                        "concept_count": len(concept_map),
                    },
                    run=lambda: generate_saint_concept_embeddings(
                        job,
                        concepts_data=concepts_data,
                        concept_map=concept_map,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda result: {
                        "embedding_count": len(result) if isinstance(result, list) else None,
                        "embedding_dim": len(result[0])
                        if isinstance(result, list) and result and isinstance(result[0], list)
                        else None,
                    },
                )

                await self._trace_step(
                    job,
                    step_id="theory_generation",
                    label="Generate theory snippets",
                    input_payload={
                        "concept_count": len(concepts_data),
                    },
                    run=lambda: generate_concept_theories(
                        job,
                        concepts_data=concepts_data,
                        generate_theory=bindings.generate_theory,
                        persist_job_state=self._persist_job_state,
                    ),
                    output_payload=lambda _result: {
                        "concept_count": len(concepts_data),
                        "theory_count": sum(
                            1 for item in concepts_data.values() if item.get("theory")
                        ),
                    },
                )

                job.result = await self._trace_step(
                    job,
                    step_id="result_assembly",
                    label="Assemble final API payload",
                    input_payload={
                        "concept_count": len(concept_map),
                        "prereq_edge_count": len(prereq_edges),
                        "quality_passed": quality_report["passed"],
                    },
                    run=lambda: asyncio.sleep(
                        0,
                        result=assemble_pipeline_result(
                            concepts_data=concepts_data,
                            concept_map=concept_map,
                            prereq_edges=prereq_edges,
                            concept_embeddings=concept_embeddings_list,
                            stats=stats,
                        ),
                    ),
                    output_payload=lambda result: {
                        "concept_count": len(result.get("concept_map", {})),
                        "prereq_edge_count": len(result.get("prereq_edges", [])),
                        "graph_node_count": len(result.get("graph", {}).get("nodes", [])),
                        "graph_edge_count": len(result.get("graph", {}).get("edges", [])),
                    },
                )
                job.result["quality_report"] = quality_report
                job.result["page_batch_size"] = job.page_batch_size
                job.result["batch_count"] = job.batch_count
                job.result["failed_batches"] = outcome.failed_batches
                job.result["warnings"] = outcome.warnings
                job.result["partial_success"] = job.partial_success
                populate_job_metrics_from_result(job)

                # Checkpoint 7: right before finalization
                await self._check_cancelled(job)

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
        except JobCancelledError as exc:
            await self._persist_cancelled(job, exc)
            return
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
