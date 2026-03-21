"""Composition layer for running the unified content pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ....config import get_settings
from ..domain.jobs import PipelineJob, PipelineStatus
from ..infrastructure.runtime import (
    calculate_file_hash,
    get_content_processor_bindings,
    get_s3_client,
)
from .stages.cache_restore import (
    try_restore_completed_job_from_mongo,
    try_restore_completed_job_from_s3,
)
from .stages.concept_extraction import extract_concepts_from_chunks
from .stages.concept_merge import merge_duplicate_concepts
from .stages.document_loading import load_document_chunks
from .stages.embedding import compute_concept_embeddings, resolve_embedding_settings
from .stages.enrichment import (
    generate_concept_theories,
    generate_saint_concept_embeddings,
)
from .stages.finalization import (
    complete_pipeline_job,
    persist_terminal_failure,
    upload_result_cache,
)
from .stages.graph_building import build_knowledge_graph
from .stages.graph_optimization import optimize_graph
from .stages.prerequisite_ranking import rank_candidate_prerequisites
from .stages.relation_verification import verify_candidate_relations
from .stages.result_assembly import (
    assemble_pipeline_result,
    serialize_concepts,
    serialize_prerequisite_edges,
)


PersistJobStateFn = Callable[[PipelineJob, PipelineStatus, str, float], Awaitable[None]]
SaveJobFn = Callable[[PipelineJob], Awaitable[bool]]
LoadJobFn = Callable[[str], Awaitable[dict | None]]


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


class PipelineRunner:
    """Runs the content pipeline by composing extracted stages."""

    def __init__(
        self,
        *,
        load_job: LoadJobFn,
        save_job: SaveJobFn,
        persist_job_state: PersistJobStateFn,
    ) -> None:
        self._load_job = load_job
        self._save_job = save_job
        self._persist_job_state = persist_job_state

    async def run(
        self,
        job: PipelineJob,
        *,
        file_path: str,
        prs_threshold: float,
        min_confidence: float,
        apply_reduction: bool,
    ) -> None:
        try:
            bindings = get_content_processor_bindings()

            if await try_restore_completed_job_from_mongo(
                job,
                load_job=self._load_job,
                populate_metrics=populate_job_metrics_from_result,
            ):
                return

            settings = get_settings()
            s3_client = get_s3_client()
            bucket_name = settings.s3_bucket_name
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
                return

            chunks = await load_document_chunks(
                job,
                file_path=file_path,
                load_and_chunk=bindings.file_loader_factory.load_and_chunk,
                persist_job_state=self._persist_job_state,
            )

            llm_client = bindings.llm_factory(temperature=0.1)
            extraction_chain = bindings.extraction_chain_cls(client=llm_client)
            all_concepts = await extract_concepts_from_chunks(
                job,
                chunks=chunks,
                extraction_chain=extraction_chain,
                postprocess_concepts=bindings.postprocess_concepts,
                persist_job_state=self._persist_job_state,
            )

            model_name, batch_size = resolve_embedding_settings()
            await compute_concept_embeddings(
                job,
                concepts=all_concepts,
                embedding_client_factory=bindings.embedding_client_cls,
                compute_embedding_for_concepts=bindings.compute_embedding_for_concepts,
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

            candidate_pairs = await rank_candidate_prerequisites(
                job,
                concepts=all_concepts,
                prs_threshold=prs_threshold,
                rank_prerequisites=bindings.rank_prerequisites,
                persist_job_state=self._persist_job_state,
            )

            verified = await verify_candidate_relations(
                job,
                concepts=all_concepts,
                candidate_pairs=candidate_pairs,
                min_confidence=min_confidence,
                verify_relations_batch=extraction_chain.verify_relations_batch,
                persist_job_state=self._persist_job_state,
            )

            graph, graph_build_stats = await build_knowledge_graph(
                job,
                concepts=all_concepts,
                verified_relations=verified,
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
                text_model_factory=bindings.saint_text_model_factory,
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
        except Exception as exc:
            import traceback

            traceback.print_exc()
            await persist_terminal_failure(
                job,
                error=exc,
                save_job=self._save_job,
            )
