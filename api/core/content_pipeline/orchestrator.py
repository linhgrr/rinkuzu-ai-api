"""
content_pipeline.py — Content processor integration
"""

import sys
import uuid
import time
import asyncio
import hashlib
import json
import boto3
from botocore.client import Config
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from .. import mongo_store
from ...config import get_settings
from .domain.jobs import PipelineJob, PipelineStatus

CONTENT_PROCESSOR_SRC = str(
    Path(__file__).resolve().parents[3] / "content-processor" / "src"
)
if CONTENT_PROCESSOR_SRC not in sys.path:
    sys.path.insert(0, CONTENT_PROCESSOR_SRC)


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


async def _persist_job_state(
    job: PipelineJob,
    status: PipelineStatus,
    step: str,
    progress: float,
) -> None:
    job.status = status
    job.current_step = step
    job.progress = progress
    saved = await mongo_store.save_pipeline_job(job)
    if not saved:
        raise RuntimeError(
            f"Failed to persist pipeline job {job.job_id} at status={status.value}"
        )


def get_s3_client():
    settings = get_settings()
    if not settings.s3_available:
        return None
        
    return boto3.client(
        's3',
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=Config(s3={'addressing_style': 'path'})
    )


def calculate_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _try_import_content_processor():
    try:
        from processors.factory import FileLoaderFactory
        from processors.chunkers.text_chunker import TextChunker
        from llm.extract_chain import ExtractionChain
        from llm.postprocess import postprocess_concepts
        from embed.embedding_client import EmbeddingClient
        from embed.embeddings import compute_embedding_for_concepts
        from embed.prereq_ranking import rank_prerequisites
        from merge.name_merge import merge_by_name
        from graph.builder import KnowledgeGraphBuilder
        from graph.cycle_removal import make_dag_with_llm
        from graph.reduction import apply_transitive_reduction
        return True, None
    except ImportError as e:
        import traceback
        err = f"{e}\n\nsys.path: {sys.path}\n\nTraceback:\n{traceback.format_exc()}"
        logger.warning(f"Content processor not available: {err}")
        return False, str(e)


_cp_result = _try_import_content_processor()
CONTENT_PROCESSOR_AVAILABLE = _cp_result[0]
CONTENT_PROCESSOR_ERROR = _cp_result[1]


async def process_pdf(
    file_path: str,
    subject_id: Optional[str] = None,
    prs_threshold: float = 0.75,
    min_confidence: float = 0.6,
    apply_reduction: bool = True,
    user_id: Optional[str] = None,
    background_tasks: Optional[Any] = None,
) -> PipelineJob:
    job_id = str(uuid.uuid4())[:8]
    filename = Path(file_path).name
    if not subject_id:
        subject_id = Path(file_path).stem

    job = PipelineJob(
        job_id=job_id,
        filename=filename,
        subject_id=subject_id,
        user_id=user_id,
    )

    if not CONTENT_PROCESSOR_AVAILABLE:
        job.status = PipelineStatus.FAILED
        job.error_message = (
            "Content processor modules not available. "
            f"Expected at: {CONTENT_PROCESSOR_SRC}"
        )
        return job

    persisted = await mongo_store.save_pipeline_job(job)
    if not persisted:
        raise RuntimeError(f"Failed to persist pipeline job {job.job_id}")

    await _persist_job_state(
        job,
        PipelineStatus.QUEUED,
        "Queued for processing",
        0.01,
    )

    if background_tasks:
        background_tasks.add_task(
            _run_pipeline_and_cleanup, job, file_path, prs_threshold, min_confidence, apply_reduction
        )
    else:
        asyncio.create_task(_run_pipeline_and_cleanup(job, file_path, prs_threshold, min_confidence, apply_reduction))

    return job

async def _run_pipeline_and_cleanup(job, file_path, *args, **kwargs):
    try:
        await _run_pipeline(job, file_path, *args, **kwargs)
    finally:
        try:
            p = Path(file_path)
            if p.exists():
                p.unlink()
        except:
            pass


async def _run_pipeline(
    job: PipelineJob,
    file_path: str,
    prs_threshold: float,
    min_confidence: float,
    apply_reduction: bool,
):
    try:
        from processors.factory import FileLoaderFactory
        from processors.chunkers.text_chunker import TextChunker
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

        # Check MongoDB first
        mongo_doc = await mongo_store.load_pipeline_job(job.job_id)
        if (
            mongo_doc
            and mongo_doc.get("status") == PipelineStatus.COMPLETED.value
            and mongo_doc.get("result")
        ):
            job.filename = mongo_doc.get("filename", job.filename)
            job.subject_id = mongo_doc.get("subject_id", job.subject_id)
            job.total_chunks = int(mongo_doc.get("total_chunks", 0) or 0)
            job.result = mongo_doc["result"]
            job.concepts_extracted = int(mongo_doc.get("concepts_extracted", 0) or 0)
            job.concepts_after_merge = int(mongo_doc.get("concepts_after_merge", 0) or 0)
            job.relations_verified = int(mongo_doc.get("relations_verified", 0) or 0)
            job.graph_stats = (
                mongo_doc.get("graph_stats")
                if isinstance(mongo_doc.get("graph_stats"), dict)
                else {}
            )
            _populate_job_metrics_from_result(job)
            job.status = PipelineStatus.COMPLETED
            job.current_step = "Loaded from MongoDB"
            job.progress = 1.0
            job.completed_at = mongo_doc.get("completed_at", time.time())
            logger.info(f"[Pipeline] Job {job.job_id} restored from MongoDB")
            return

        # Check S3 Cache
        file_hash = calculate_file_hash(file_path)
        s3_client = get_s3_client()
        settings = get_settings()
        bucket_name = settings.s3_bucket_name
        cache_key = f"cache/{file_hash}.json"
        
        if s3_client and bucket_name:
            job.status = PipelineStatus.LOADING
            job.current_step = "Kiểm tra cache trên S3..."
            job.progress = 0.02
            try:
                response = await loop.run_in_executor(
                    None, 
                    lambda: s3_client.get_object(Bucket=bucket_name, Key=cache_key)
                )
                cache_content = response['Body'].read().decode('utf-8')
                job.result = json.loads(cache_content)
                _populate_job_metrics_from_result(job)
                job.status = PipelineStatus.COMPLETED
                job.current_step = "Loaded from S3 cache"
                job.progress = 1.0
                job.completed_at = time.time()
                logger.info(f"[Pipeline] Job {job.job_id} loaded from S3 cache {cache_key}")
                
                saved = await mongo_store.save_pipeline_job(job)
                if not saved:
                    raise RuntimeError("Failed to persist S3-cached pipeline result to MongoDB")
                return
            except Exception:
                logger.debug(f"[Pipeline] Cache miss: {cache_key}")

        # Step 1: Load & Chunk PDF
        await _persist_job_state(job, PipelineStatus.LOADING, "Loading PDF...", 0.05)
        chunks = await loop.run_in_executor(
            None, FileLoaderFactory.load_and_chunk, file_path, job.subject_id
        )
        job.total_chunks = len(chunks)
        await _persist_job_state(job, PipelineStatus.LOADING, "Loading PDF...", 0.10)

        # Step 2: Extract concepts via LLM
        await _persist_job_state(job, PipelineStatus.EXTRACTING, "Extracting concepts with LLM...", 0.15)

        llm_client = get_llm(temperature=0.1)
        extraction_chain = ExtractionChain(client=llm_client)

        chunk_texts = [c.page_content for c in chunks]
        extractions = await loop.run_in_executor(
            None,
            extraction_chain.extract_from_batch,
            chunk_texts,
            job.subject_id,
        )

        all_concepts = []
        for ext in extractions:
            if ext and hasattr(ext, "concepts"):
                all_concepts.extend(ext.concepts)

        all_concepts = postprocess_concepts(all_concepts)
        job.concepts_extracted = len(all_concepts)
        job.partial_graph = {
            "nodes": [{"id": getattr(c, "concept_id", ""), "name": getattr(c, "name", "")} for c in all_concepts],
            "edges": []
        }
        await _persist_job_state(job, PipelineStatus.EXTRACTING, "Extracting concepts with LLM...", 0.30)

        # Step 3: Compute embeddings
        await _persist_job_state(job, PipelineStatus.EMBEDDING, "Computing embeddings...", 0.35)

        # Assuming config.settings from content_processor has defaults
        try:
            from config import settings as cp_settings
            model_name = cp_settings.embedding_model
            batch_size = cp_settings.embedding_batch_size
        except ImportError:
            model_name = "keepitreal/vietnamese-sbert"
            batch_size = 32
            
        embed_client = EmbeddingClient(model_name=model_name, batch_size=batch_size)
        await loop.run_in_executor(
            None, compute_embedding_for_concepts, all_concepts, embed_client
        )
        await _persist_job_state(job, PipelineStatus.EMBEDDING, "Computing embeddings...", 0.45)

        # Step 4: Merge & Deduplicate
        await _persist_job_state(job, PipelineStatus.MERGING, "Merging duplicate concepts...", 0.50)

        all_concepts = await loop.run_in_executor(None, merge_by_name, all_concepts)
        job.concepts_after_merge = len(all_concepts)
        job.partial_graph = {
            "nodes": [{"id": getattr(c, "concept_id", ""), "name": getattr(c, "name", "")} for c in all_concepts],
            "edges": []
        }
        await _persist_job_state(job, PipelineStatus.MERGING, "Merging duplicate concepts...", 0.55)

        # Step 5: Prerequisite ranking
        await _persist_job_state(job, PipelineStatus.RANKING, "Ranking prerequisites...", 0.60)

        candidate_pairs = await loop.run_in_executor(
            None, rank_prerequisites, all_concepts, prs_threshold
        )
        await _persist_job_state(job, PipelineStatus.RANKING, "Ranking prerequisites...", 0.65)

        # Step 6: Verify relations via LLM
        await _persist_job_state(job, PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.70)

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
        await _persist_job_state(job, PipelineStatus.VERIFYING, "Verifying relations with LLM...", 0.80)

        # Step 7: Build knowledge graph
        await _persist_job_state(job, PipelineStatus.BUILDING_GRAPH, "Building knowledge graph...", 0.85)

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
        await _persist_job_state(job, PipelineStatus.OPTIMIZING, "Removing cycles, building DAG...", 0.90)

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

        await _persist_job_state(job, PipelineStatus.OPTIMIZING, "Removing cycles, building DAG...", 0.95)

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
        await _persist_job_state(
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
        await _persist_job_state(
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
        await _persist_job_state(job, PipelineStatus.COMPLETED, "Processing complete!", 1.0)

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
