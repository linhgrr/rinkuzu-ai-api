"""
content_pipeline.py — Content processor integration
Imports from content-processor via sys.path to reuse:
  PDF loading, text chunking, LLM concept extraction,
  embedding, merge/dedup, prerequisite ranking, DAG construction.
"""

import sys
import os
import uuid
import time
import asyncio
import hashlib
import json
import boto3
from botocore.client import Config
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from . import mongo_store

# Add content-processor src to sys.path
# content_pipeline.py is at full-demo/api/core/ → parents[2] = full-demo/
CONTENT_PROCESSOR_SRC = str(
    Path(__file__).resolve().parents[2] / "content-processor" / "src"
)
if CONTENT_PROCESSOR_SRC not in sys.path:
    sys.path.insert(0, CONTENT_PROCESSOR_SRC)


class PipelineStatus(str, Enum):
    PENDING = "pending"
    LOADING = "loading"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    MERGING = "merging"
    RANKING = "ranking"
    VERIFYING = "verifying"
    BUILDING_GRAPH = "building_graph"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineJob:
    job_id: str
    filename: str
    subject_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    current_step: str = ""
    progress: float = 0.0
    total_chunks: int = 0
    concepts_extracted: int = 0
    concepts_after_merge: int = 0
    relations_verified: int = 0
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    partial_graph: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# In-memory job store
_jobs: Dict[str, PipelineJob] = {}


def get_job(job_id: str) -> Optional[PipelineJob]:
    return _jobs.get(job_id)


def _populate_job_metrics_from_result(job: PipelineJob) -> None:
    """Hydrate summary counters from job.result when loading cached/persisted data."""
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

    # Keep concepts_extracted if already computed in runtime; otherwise infer.
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
    """Keep only valid PREREQUISITE relations to existing concept IDs."""
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


def get_s3_client():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    access_key = os.getenv("S3_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
    
    if not all([endpoint_url, access_key, secret_key]):
        return None
        
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={'addressing_style': 'path'})
    )


def calculate_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _try_import_content_processor():
    """Try to import content-processor modules. Returns (available, error_msg)."""
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
        print(f"Content processor not available: {err}")
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
) -> PipelineJob:
    """
    Full pipeline: PDF -> concepts -> prerequisites -> DAG.
    Runs in background, returns job immediately.
    """
    job_id = str(uuid.uuid4())[:8]
    filename = Path(file_path).name
    if not subject_id:
        subject_id = Path(file_path).stem

    job = PipelineJob(
        job_id=job_id,
        filename=filename,
        subject_id=subject_id,
    )
    _jobs[job_id] = job

    if not CONTENT_PROCESSOR_AVAILABLE:
        job.status = PipelineStatus.FAILED
        job.error_message = (
            "Content processor modules not available. "
            f"Expected at: {CONTENT_PROCESSOR_SRC}"
        )
        return job

    # Run pipeline in background
    asyncio.create_task(_run_pipeline(job, file_path, prs_threshold, min_confidence, apply_reduction))
    return job


async def _run_pipeline(
    job: PipelineJob,
    file_path: str,
    prs_threshold: float,
    min_confidence: float,
    apply_reduction: bool,
):
    """Execute the full 10-step content processing pipeline."""
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
        from config import settings

        loop = asyncio.get_event_loop()

        # --- Check MongoDB first ---
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
            print(f"[Pipeline] Job {job.job_id} restored from MongoDB")
            return

        # --- Check S3 Cache ---
        file_hash = calculate_file_hash(file_path)
        s3_client = get_s3_client()
        bucket_name = os.getenv("S3_BUCKET_NAME")
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
                print(f"[Pipeline] Job {job.job_id} loaded from S3 cache {cache_key}")
                # Also save to MongoDB so it shows up on the Dashboard
                saved = await mongo_store.save_pipeline_job(job)
                if not saved:
                    raise RuntimeError("Failed to persist S3-cached pipeline result to MongoDB")
                return
            except Exception as e:
                # Not found or error loading, continue normal pipeline
                print(f"[Pipeline] Cache miss: {cache_key}")

        # Step 1: Load & Chunk PDF
        job.status = PipelineStatus.LOADING
        job.current_step = "Loading PDF..."
        job.progress = 0.05
        chunks = await loop.run_in_executor(
            None, FileLoaderFactory.load_and_chunk, file_path, job.subject_id
        )
        job.total_chunks = len(chunks)
        job.progress = 0.10

        # Step 2: Extract concepts via LLM
        job.status = PipelineStatus.EXTRACTING
        job.current_step = "Extracting concepts with LLM..."
        job.progress = 0.15

        llm_client = get_llm(
            temperature=0.1,
        )
        extraction_chain = ExtractionChain(client=llm_client)

        chunk_texts = [c.page_content for c in chunks]
        extractions = await loop.run_in_executor(
            None,
            extraction_chain.extract_from_batch,
            chunk_texts,
            job.subject_id,
        )

        # Flatten concepts
        all_concepts = []
        for ext in extractions:
            if ext and hasattr(ext, "concepts"):
                all_concepts.extend(ext.concepts)

        # Postprocess
        all_concepts = postprocess_concepts(all_concepts)
        job.concepts_extracted = len(all_concepts)
        job.partial_graph = {
            "nodes": [{"id": getattr(c, "concept_id", ""), "name": getattr(c, "name", "")} for c in all_concepts],
            "edges": []
        }
        job.progress = 0.30

        # Step 3: Compute embeddings
        job.status = PipelineStatus.EMBEDDING
        job.current_step = "Computing embeddings..."
        job.progress = 0.35

        embed_client = EmbeddingClient(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        await loop.run_in_executor(
            None, compute_embedding_for_concepts, all_concepts, embed_client
        )
        job.progress = 0.45

        # Step 4: Merge & Deduplicate
        job.status = PipelineStatus.MERGING
        job.current_step = "Merging duplicate concepts..."
        job.progress = 0.50

        all_concepts = await loop.run_in_executor(None, merge_by_name, all_concepts)
        job.concepts_after_merge = len(all_concepts)
        job.partial_graph = {
            "nodes": [{"id": getattr(c, "concept_id", ""), "name": getattr(c, "name", "")} for c in all_concepts],
            "edges": []
        }
        job.progress = 0.55

        # Step 5: Prerequisite ranking
        job.status = PipelineStatus.RANKING
        job.current_step = "Ranking prerequisites..."
        job.progress = 0.60

        candidate_pairs = await loop.run_in_executor(
            None, rank_prerequisites, all_concepts, prs_threshold
        )
        job.progress = 0.65

        # Step 6: Verify relations via LLM
        job.status = PipelineStatus.VERIFYING
        job.current_step = "Verifying relations with LLM..."
        job.progress = 0.70

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
        job.progress = 0.80

        # Step 7: Build knowledge graph
        job.status = PipelineStatus.BUILDING_GRAPH
        job.current_step = "Building knowledge graph..."
        job.progress = 0.85

        concept_ids_set = {
            str(getattr(c, "concept_id", "")).strip()
            for c in all_concepts
            if getattr(c, "concept_id", None)
        }
        extracted_relation_count, dropped_relation_count = _sanitize_concept_relations(all_concepts)
        if dropped_relation_count:
            print(
                f"[Pipeline] Dropped {dropped_relation_count} invalid extracted relations "
                f"(target missing/self-loop/non-PREREQUISITE)"
            )

        builder = KnowledgeGraphBuilder(subject_id=job.subject_id)
        builder.add_concepts(all_concepts)
        graph = builder.get_graph()

        # Guardrail: remove any placeholder/invalid nodes or non-prerequisite edges.
        edges_to_remove = []
        for src, tgt, data in list(graph.edges(data=True)):
            rel_type = str(data.get("relation_type", "PREREQUISITE")).upper()
            if rel_type != "PREREQUISITE" or src not in concept_ids_set or tgt not in concept_ids_set:
                edges_to_remove.append((src, tgt))
        if edges_to_remove:
            graph.remove_edges_from(edges_to_remove)
            print(f"[Pipeline] Removed {len(edges_to_remove)} invalid graph edges before optimization")

        orphan_nodes = [node_id for node_id in list(graph.nodes()) if node_id not in concept_ids_set]
        if orphan_nodes:
            graph.remove_nodes_from(orphan_nodes)
            print(f"[Pipeline] Removed {len(orphan_nodes)} placeholder nodes not in concept list")

        existing_edges = set(graph.edges())
        print(f"[Pipeline] Added {extracted_relation_count} relations from LLM extraction")

        # --- Add embedding-verified relations (Steps 5-6) ---
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

        # Final cleanup after adding verified edges
        edges_to_remove = []
        for src, tgt, data in list(graph.edges(data=True)):
            rel_type = str(data.get("relation_type", "PREREQUISITE")).upper()
            if rel_type != "PREREQUISITE" or src not in concept_ids_set or tgt not in concept_ids_set:
                edges_to_remove.append((src, tgt))
        if edges_to_remove:
            graph.remove_edges_from(edges_to_remove)
            print(f"[Pipeline] Removed {len(edges_to_remove)} invalid graph edges after verification")

        orphan_nodes = [node_id for node_id in list(graph.nodes()) if node_id not in concept_ids_set]
        if orphan_nodes:
            graph.remove_nodes_from(orphan_nodes)
            print(f"[Pipeline] Removed {len(orphan_nodes)} placeholder nodes after verification")

        print(f"[Pipeline] Added {verified_relation_count} relations from embedding verification")
        print(f"[Pipeline] Total graph edges: {graph.number_of_edges()}")

        def _update_partial_from_nx(nx_graph, c_list):
            c_map = {getattr(c, "concept_id", ""): getattr(c, "name", "") for c in c_list}
            job.partial_graph = {
                "nodes": [{"id": n, "name": c_map.get(n, str(n))} for n in nx_graph.nodes()],
                "edges": [{"source": u, "target": v} for u, v, _ in nx_graph.edges(data=True)]
            }
        
        _update_partial_from_nx(graph, all_concepts)

        # Step 8: Make DAG (remove cycles)
        job.status = PipelineStatus.OPTIMIZING
        job.current_step = "Removing cycles, building DAG..."
        job.progress = 0.90

        import networkx as nx
        if not nx.is_directed_acyclic_graph(graph):
            graph, cycle_stats = await loop.run_in_executor(
                None, make_dag_with_llm, graph
            )
            _update_partial_from_nx(graph, all_concepts)

        # Step 9: Transitive reduction
        if apply_reduction:
            graph = await loop.run_in_executor(
                None, apply_transitive_reduction, graph
            )
            _update_partial_from_nx(graph, all_concepts)

        job.progress = 0.95

        # Build result
        stats = builder.get_stats()
        stats["num_nodes"] = graph.number_of_nodes()
        stats["num_edges"] = graph.number_of_edges()
        stats["is_dag"] = nx.is_directed_acyclic_graph(graph)
        stats["relations_from_extraction"] = extracted_relation_count
        stats["relations_from_verification"] = verified_relation_count
        stats["relations_verified"] = job.relations_verified
        job.graph_stats = stats

        # Convert to format usable by SessionManager
        concepts_data = {}
        concept_map = {}
        for i, concept in enumerate(all_concepts):
            cid = concept.concept_id
            if cid in concept_map:
                raise ValueError(f"Duplicate concept_id after merge: {cid}")
            concept_map[cid] = i
            # Save LLM-extracted relations alongside concept data
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
        job.current_step = "Generating concept embeddings for SAINT..."
        job.progress = 0.92
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
            print(f"[Pipeline] ✓ Generated embeddings for {len(ordered_texts)} concepts ({concept_embeddings.shape})")
        except Exception as e:
            print(f"[Pipeline] ⚠ Could not generate embeddings: {e}")
            concept_embeddings_list = None

        # --- Generate concept theories ---
        job.current_step = "Generating concept theories..."
        job.progress = 0.93
        try:
            from .exercise_gen import generate_theory
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
                print(f"[Pipeline] Generating theory for {len(tasks)} concepts...")
                results = await asyncio.gather(*tasks)
                for cid, theory in results:
                    concepts_data[cid]["theory"] = theory
                print("[Pipeline] ✓ Theory generation complete")
        except Exception as e:
            print(f"[Pipeline] ⚠ Failed to pre-generate theory: {e}")

        # Graph nodes/edges for frontend
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

        job.status = PipelineStatus.COMPLETED
        job.current_step = "Processing complete!"
        job.progress = 1.0
        job.completed_at = time.time()

        # --- Save to MongoDB ---
        saved = await mongo_store.save_pipeline_job(job)
        if not saved:
            raise RuntimeError("Failed to persist pipeline result to MongoDB")

        # --- Save to S3 Cache ---
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
                print(f"[Pipeline] Uploaded result to S3 cache {cache_key}")
            except Exception as e:
                print(f"[Pipeline] Failed to save S3 cache: {e}")

        print(f"[Pipeline] Job {job.job_id} completed: "
              f"{job.concepts_after_merge} concepts, "
              f"{len(prereq_edges)} prerequisite edges")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = PipelineStatus.FAILED
        job.error_message = str(e)
        job.current_step = f"Error: {e}"
        print(f"[Pipeline] Job {job.job_id} failed: {e}")
