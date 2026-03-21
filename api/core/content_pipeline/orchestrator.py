"""Legacy content processor orchestration entrypoints."""

import time
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from .. import mongo_store
from .application.pipeline_service import PipelineService
from .application.stages.cache_restore import (
    try_restore_completed_job_from_mongo,
    try_restore_completed_job_from_s3,
)
from .application.stages.concept_extraction import extract_concepts_from_chunks
from .application.stages.document_loading import load_document_chunks
from .application.stages.embedding import (
    compute_concept_embeddings,
    resolve_embedding_settings,
)
from .domain.jobs import PipelineJob, PipelineStatus
from .infrastructure.runtime import (
    CONTENT_PROCESSOR_AVAILABLE,
    CONTENT_PROCESSOR_ERROR,
    CONTENT_PROCESSOR_SRC,
    calculate_file_hash,
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


def _sanitize_concept_relations(all_concepts: List[Any]) -> Tuple[int, int]:
    concept_ids = {
        str(getattr(c, "concept_id", "")).strip()
        for c in all_concepts
        if getattr(c, "concept_id", None)
    }
    kept = 0
    dropped = 0

    for concept in all_concepts:
        source_id = str(getattr(concept, "concept_id", "")).strip()
        seen_targets = set()
        cleaned_relations = []

        for rel in getattr(concept, "relations", []) or []:
            rel_type = str(getattr(rel, "type", "")).strip().upper()
            target_id = str(getattr(rel, "target_id", "")).strip()

            if rel_type != "PREREQUISITE" or not target_id or target_id == source_id:
                dropped += 1
                continue
            if target_id not in concept_ids:
                dropped += 1
                continue
            if target_id in seen_targets:
                continue

            seen_targets.add(target_id)
            cleaned_relations.append(rel)
            kept += 1

        concept.relations = cleaned_relations

    return kept, dropped


def _build_partial_graph(graph, all_concepts: List[Any]) -> Dict[str, Any]:
    concept_map = {
        getattr(concept, "concept_id", ""): getattr(concept, "name", "")
        for concept in all_concepts
    }
    return {
        "nodes": [
            {"id": node_id, "name": concept_map.get(node_id, str(node_id))}
            for node_id in graph.nodes()
        ],
        "edges": [{"source": src, "target": tgt} for src, tgt in graph.edges()],
    }


def _remove_invalid_graph_members(graph, concept_ids: set[str]) -> None:
    edges_to_remove = []
    for src, tgt, data in list(graph.edges(data=True)):
        rel_type = str(data.get("relation_type", "PREREQUISITE")).upper()
        if rel_type != "PREREQUISITE" or src not in concept_ids or tgt not in concept_ids:
            edges_to_remove.append((src, tgt))
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)

    orphan_nodes = [node_id for node_id in list(graph.nodes()) if node_id not in concept_ids]
    if orphan_nodes:
        graph.remove_nodes_from(orphan_nodes)


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
        from processors.factory import FileLoaderFactory
        from llm.extract_chain import ExtractionChain
        from llm.postprocess import postprocess_concepts
        from llm import get_llm
        from embed.embedding_client import EmbeddingClient
        from embed.embeddings import compute_embedding_for_concepts
        from embed.prereq_ranking import rank_prerequisites
        from merge.name_merge import merge_by_name
        from graph.builder import KnowledgeGraphBuilder
        from graph.cycle_removal import make_dag_with_llm
        from graph.reduction import apply_transitive_reduction
        
        # from config import settings as cp_settings (if needed inside content_processor)

        loop = asyncio.get_event_loop()

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
            load_and_chunk=FileLoaderFactory.load_and_chunk,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        llm_client = get_llm(temperature=0.1)
        extraction_chain = ExtractionChain(client=llm_client)
        all_concepts = await extract_concepts_from_chunks(
            job,
            chunks=chunks,
            extraction_chain=extraction_chain,
            postprocess_concepts=postprocess_concepts,
            persist_job_state=get_pipeline_service().persist_job_state,
        )

        model_name, batch_size = resolve_embedding_settings()
        await compute_concept_embeddings(
            job,
            concepts=all_concepts,
            embedding_client_factory=EmbeddingClient,
            compute_embedding_for_concepts=compute_embedding_for_concepts,
            persist_job_state=get_pipeline_service().persist_job_state,
            model_name=model_name,
            batch_size=batch_size,
        )

        # Step 4: Merge & Deduplicate
        await get_pipeline_service().persist_job_state(job, PipelineStatus.MERGING, "Merging duplicate concepts...", 0.50)

        all_concepts = await loop.run_in_executor(None, merge_by_name, all_concepts)
        job.concepts_after_merge = len(all_concepts)
        job.partial_graph = {
            "nodes": [{"id": getattr(c, "concept_id", ""), "name": getattr(c, "name", "")} for c in all_concepts],
            "edges": []
        }
        await get_pipeline_service().persist_job_state(job, PipelineStatus.MERGING, "Merging duplicate concepts...", 0.55)

        # Step 5: Prerequisite ranking
        await get_pipeline_service().persist_job_state(job, PipelineStatus.RANKING, "Ranking prerequisites...", 0.60)

        candidate_pairs = await loop.run_in_executor(
            None, rank_prerequisites, all_concepts, prs_threshold
        )
        await get_pipeline_service().persist_job_state(job, PipelineStatus.RANKING, "Ranking prerequisites...", 0.65)

        # Step 6: Verify relations via LLM
        await get_pipeline_service().persist_job_state(job, PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.70)

        concept_name_map = {c.concept_id: c.name for c in all_concepts}
        pairs_to_verify = [
            (concept_name_map.get(a, a), concept_name_map.get(b, b))
            for a, b in candidate_pairs
        ]

        verified = []
        if pairs_to_verify:
            verifications = await loop.run_in_executor(
                None, extraction_chain.verify_relations_batch, pairs_to_verify
            )
            for (cid_a, cid_b), ev in zip(candidate_pairs, verifications):
                if ev and ev.has_relation and ev.confidence >= min_confidence:
                    verified.append((cid_a, cid_b, ev))

        job.relations_verified = len(verified)
        await get_pipeline_service().persist_job_state(job, PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.80)

        # Step 7: Build knowledge graph
        await get_pipeline_service().persist_job_state(job, PipelineStatus.BUILDING_GRAPH, "Building knowledge graph...", 0.85)

        concept_ids_set = {
            str(getattr(c, "concept_id", "")).strip()
            for c in all_concepts
            if getattr(c, "concept_id", None)
        }
        extracted_relation_count, dropped_relation_count = _sanitize_concept_relations(all_concepts)
        if dropped_relation_count:
            logger.info(
                f"[Pipeline] Dropped {dropped_relation_count} invalid extracted relations"
            )

        builder = KnowledgeGraphBuilder(subject_id=job.subject_id)
        builder.add_concepts(all_concepts)
        graph = builder.get_graph()
        _remove_invalid_graph_members(graph, concept_ids_set)

        existing_edges = set(graph.edges())
        logger.debug(f"[Pipeline] Added {extracted_relation_count} relations from extraction")

        verified_relation_count = 0
        for cid_a, cid_b, ev in verified:
            if cid_a not in concept_ids_set or cid_b not in concept_ids_set:
                continue
            if hasattr(ev, "direction"):
                if ev.direction == "A_to_B":
                    edge = (cid_a, cid_b)
                    if edge not in existing_edges:
                        builder.add_relation(cid_a, cid_b, "PREREQUISITE")
                        existing_edges.add(edge)
                        verified_relation_count += 1
                elif ev.direction == "B_to_A":
                    edge = (cid_b, cid_a)
                    if edge not in existing_edges:
                        builder.add_relation(cid_b, cid_a, "PREREQUISITE")
                        existing_edges.add(edge)
                        verified_relation_count += 1

        _remove_invalid_graph_members(graph, concept_ids_set)
        job.partial_graph = _build_partial_graph(graph, all_concepts)

        # Step 8: Make DAG
        await get_pipeline_service().persist_job_state(job, PipelineStatus.OPTIMIZING, "Removing cycles, building DAG...", 0.90)

        import networkx as nx
        if not nx.is_directed_acyclic_graph(graph):
            graph, cycle_stats = await loop.run_in_executor(
                None, make_dag_with_llm, graph
            )
            job.partial_graph = _build_partial_graph(graph, all_concepts)

        # Step 9: Transitive reduction
        if apply_reduction:
            graph = await loop.run_in_executor(
                None, apply_transitive_reduction, graph
            )
            job.partial_graph = _build_partial_graph(graph, all_concepts)

        await get_pipeline_service().persist_job_state(job, PipelineStatus.OPTIMIZING, "Removing cycles, building DAG...", 0.95)

        stats = builder.get_stats()
        stats["num_nodes"] = graph.number_of_nodes()
        stats["num_edges"] = graph.number_of_edges()
        stats["is_dag"] = nx.is_directed_acyclic_graph(graph)
        stats["relations_from_extraction"] = extracted_relation_count
        stats["relations_from_verification"] = verified_relation_count
        stats["relations_verified"] = job.relations_verified
        job.graph_stats = stats

        concepts_data = {}
        concept_map = {}
        for i, concept in enumerate(all_concepts):
            cid = concept.concept_id
            concept_map[cid] = i
            concept_relations = []
            if hasattr(concept, "relations") and concept.relations:
                for rel in concept.relations:
                    concept_relations.append({
                        "type": rel.type,
                        "target_id": rel.target_id,
                        "confidence": rel.confidence,
                        "evidence": rel.evidence,
                    })
            concepts_data[cid] = {
                "name": concept.name,
                "definition": concept.definition,
                "examples": concept.examples if hasattr(concept, "examples") else [],
                "relations": concept_relations,
            }

        prereq_edges = []
        for src, tgt, data in graph.edges(data=True):
            rel_type = data.get("relation_type", "PREREQUISITE")
            if rel_type == "PREREQUISITE" and src in concept_map and tgt in concept_map:
                prereq_edges.append({"source": src, "target": tgt})

        # --- Generate concept embeddings for SAINT ---
        await get_pipeline_service().persist_job_state(
            job,
            PipelineStatus.OPTIMIZING,
            "Generating concept embeddings for SAINT...",
            0.92,
        )
        try:
            from sentence_transformers import SentenceTransformer
            text_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            id_to_concept = {v: k for k, v in concept_map.items()}
            ordered_texts = []
            for idx in range(len(concept_map)):
                cid = id_to_concept.get(idx, str(idx))
                name = concepts_data[cid]["name"]
                definition = concepts_data[cid].get("definition", "")
                text = f"{name}: {definition}" if definition else name
                ordered_texts.append(text)
            concept_embeddings = await loop.run_in_executor(
                None,
                lambda: text_model.encode(ordered_texts, show_progress_bar=False, batch_size=32)
            )
            concept_embeddings_list = concept_embeddings.tolist()
            logger.info(f"[Pipeline] ✓ Generated embeddings for {len(ordered_texts)} concepts")
        except Exception as e:
            logger.warning(f"[Pipeline] ⚠ Could not generate embeddings: {e}")
            concept_embeddings_list = None

        # --- Generate concept theories ---
        await get_pipeline_service().persist_job_state(
            job,
            PipelineStatus.OPTIMIZING,
            "Generating concept theories...",
            0.93,
        )
        try:
            from ..exercise_gen import generate_theory
            sem = asyncio.Semaphore(5)
            
            async def _generate_theory_wrapper(cid, name, definition):
                async with sem:
                    res = await loop.run_in_executor(
                        None,
                        generate_theory, name, definition
                    )
                    return cid, res
                    
            tasks = []
            for cid, cdata in concepts_data.items():
                if "theory" not in cdata:
                    tasks.append(_generate_theory_wrapper(cid, cdata["name"], cdata.get("definition", "")))
            
            if tasks:
                logger.info(f"[Pipeline] Generating theory for {len(tasks)} concepts...")
                results = await asyncio.gather(*tasks)
                for cid, theory in results:
                    concepts_data[cid]["theory"] = theory
                logger.info("[Pipeline] ✓ Theory generation complete")
        except Exception as e:
            logger.warning(f"[Pipeline] ⚠ Failed to pre-generate theory: {e}")

        nodes = []
        for cid, idx in concept_map.items():
            nodes.append({
                "id": cid,
                "index": idx,
                "name": concepts_data[cid]["name"],
                "definition": concepts_data[cid].get("definition", ""),
            })

        job.result = {
            "concepts_data": concepts_data,
            "concept_map": concept_map,
            "prereq_edges": prereq_edges,
            "concept_embeddings": concept_embeddings_list,
            "graph": {
                "nodes": nodes,
                "edges": prereq_edges,
            },
            "stats": stats,
        }
        _populate_job_metrics_from_result(job)

        job.completed_at = time.time()
        await get_pipeline_service().persist_job_state(job, PipelineStatus.COMPLETED, "Processing complete!", 1.0)

        if s3_client and bucket_name:
            try:
                cache_data = json.dumps(job.result, ensure_ascii=False)
                await loop.run_in_executor(
                    None,
                    lambda: s3_client.put_object(
                        Bucket=bucket_name,
                        Key=cache_key,
                        Body=cache_data,
                        ContentType='application/json'
                    )
                )
                logger.info(f"[Pipeline] Uploaded result to S3 cache {cache_key}")
            except Exception as e:
                logger.warning(f"[Pipeline] Failed to save S3 cache: {e}")

        logger.info(f"[Pipeline] Job {job.job_id} completed: {job.concepts_after_merge} concepts")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.error_message = str(e)
        logger.error(f"[Pipeline] Job {job.job_id} failed: {e}")
        job.status = PipelineStatus.FAILED
        job.current_step = f"Error: {e}"
        try:
            saved = await mongo_store.save_pipeline_job(job)
            if not saved:
                logger.error(
                    f"[Pipeline] Failed to persist terminal failure state for job {job.job_id}"
                )
        except Exception as persist_exc:
            logger.error(
                f"[Pipeline] Failed to persist terminal failure state for job {job.job_id}: {persist_exc}"
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
