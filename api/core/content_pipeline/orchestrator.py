"""Legacy content processor orchestration entrypoints."""

import time
from typing import Dict, Any, Optional

from .. import mongo_store
from .application.pipeline_service import PipelineService
from .application.stages.cache_restore import (
    try_restore_completed_job_from_mongo,
    try_restore_completed_job_from_s3,
)
from .application.stages.concept_extraction import extract_concepts_from_chunks
from .application.stages.concept_merge import merge_duplicate_concepts
from .application.stages.document_loading import load_document_chunks
from .application.stages.embedding import (
    compute_concept_embeddings,
    resolve_embedding_settings,
)
from .application.stages.finalization import (
    complete_pipeline_job,
    persist_terminal_failure,
    upload_result_cache,
)
from .application.stages.enrichment import (
    generate_concept_theories,
    generate_saint_concept_embeddings,
)
from .application.stages.graph_building import build_knowledge_graph
from .application.stages.graph_optimization import optimize_graph
from .application.stages.prerequisite_ranking import rank_candidate_prerequisites
from .application.stages.relation_verification import verify_candidate_relations
from .application.stages.result_assembly import (
    assemble_pipeline_result,
    serialize_concepts,
    serialize_prerequisite_edges,
)
from .domain.jobs import PipelineJob, PipelineStatus
from .infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_SRC,
    calculate_file_hash,
    get_content_processor_bindings,
    get_s3_client,
)


def _populate_job_metrics_from_result(job: PipelineJob) -> None:
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

async def process_pdf(
    file_path: str,
    subject_id: Optional[str] = None,
    prs_threshold: float = 0.75,
    min_confidence: float = 0.6,
    apply_reduction: bool = True,
    user_id: Optional[str] = None,
    background_tasks: Optional[Any] = None,
) -> PipelineJob:
    return await get_pipeline_service().start_job(
        file_path=file_path,
        subject_id=subject_id,
        prs_threshold=prs_threshold,
        min_confidence=min_confidence,
        apply_reduction=apply_reduction,
        user_id=user_id,
        background_tasks=background_tasks,
        content_processor_available=CONTENT_PROCESSOR_AVAILABLE,
        content_processor_src=CONTENT_PROCESSOR_SRC,
    )


async def _run_pipeline(
    job: PipelineJob,
    file_path: str,
    prs_threshold: float,
    min_confidence: float,
    apply_reduction: bool,
):
    try:
        bindings = get_content_processor_bindings()

        if await try_restore_completed_job_from_mongo(
            job,
            load_job=mongo_store.load_pipeline_job,
            populate_metrics=_populate_job_metrics_from_result,
        ):
            return

        # Check S3 Cache
        s3_client = get_s3_client()
        from ...config import get_settings

        settings = get_settings()
        bucket_name = settings.s3_bucket_name
        cache_key = await try_restore_completed_job_from_s3(
            job,
            file_path=file_path,
            s3_client=s3_client,
            bucket_name=bucket_name,
            hash_file=calculate_file_hash,
            save_job=mongo_store.save_pipeline_job,
            populate_metrics=_populate_job_metrics_from_result,
        )
        if job.status == PipelineStatus.COMPLETED:
            return

        # Step 1: Load & Chunk PDF
        chunks = await load_document_chunks(
            job,
            file_path=file_path,
            load_and_chunk=bindings.file_loader_factory.load_and_chunk,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        llm_client = bindings.llm_factory(temperature=0.1)
        extraction_chain = bindings.extraction_chain_cls(client=llm_client)
        all_concepts = await extract_concepts_from_chunks(
            job,
            chunks=chunks,
            extraction_chain=extraction_chain,
            postprocess_concepts=bindings.postprocess_concepts,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        model_name, batch_size = resolve_embedding_settings()
        await compute_concept_embeddings(
            job,
            concepts=all_concepts,
            embedding_client_factory=bindings.embedding_client_cls,
            compute_embedding_for_concepts=bindings.compute_embedding_for_concepts,
            persist_job_state=get_pipeline_service().persist_job_state,
            model_name=model_name,
            batch_size=batch_size,
        )

        all_concepts = await merge_duplicate_concepts(
            job,
            concepts=all_concepts,
            merge_by_name=bindings.merge_by_name,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        candidate_pairs = await rank_candidate_prerequisites(
            job,
            concepts=all_concepts,
            prs_threshold=prs_threshold,
            rank_prerequisites=bindings.rank_prerequisites,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        verified = await verify_candidate_relations(
            job,
            concepts=all_concepts,
            candidate_pairs=candidate_pairs,
            min_confidence=min_confidence,
            verify_relations_batch=extraction_chain.verify_relations_batch,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        graph, graph_build_stats = await build_knowledge_graph(
            job,
            concepts=all_concepts,
            verified_relations=verified,
            knowledge_graph_builder_factory=bindings.knowledge_graph_builder_factory,
            persist_job_state=get_pipeline_service().persist_job_state,
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
            persist_job_state=get_pipeline_service().persist_job_state,
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
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        await generate_concept_theories(
            job,
            concepts_data=concepts_data,
            generate_theory=bindings.generate_theory,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        job.result = assemble_pipeline_result(
            concepts_data=concepts_data,
            concept_map=concept_map,
            prereq_edges=prereq_edges,
            concept_embeddings=concept_embeddings_list,
            stats=stats,
        )
        _populate_job_metrics_from_result(job)

        await complete_pipeline_job(
            job,
            persist_job_state=get_pipeline_service().persist_job_state,
        )
        await upload_result_cache(
            result=job.result,
            s3_client=s3_client,
            bucket_name=bucket_name,
            cache_key=cache_key,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        await persist_terminal_failure(
            job,
            error=e,
            save_job=mongo_store.save_pipeline_job,
        )


_pipeline_service: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService(
            save_job=mongo_store.save_pipeline_job,
            run_pipeline=_run_pipeline,
        )
    return _pipeline_service
