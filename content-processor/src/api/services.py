"""Service layer for knowledge graph pipeline processing."""

from api.models import ProcessingStatus, ProcessingResult, GraphStatsResponse
from storage.chroma_store import ConceptChromaStore
from llm.schemas import Concept, Relation
from graph.cycle_removal import make_dag_with_llm
from graph.reduction import apply_transitive_reduction
from graph.builder import KnowledgeGraphBuilder
from merge import merge_by_name, deduplicate_by_embedding
from embed.prereq_ranking import rank_prerequisites
from embed.embedding_client import EmbeddingClient
from embed.embeddings import compute_embedding_for_concepts
from llm.extract_chain import ExtractionChain
from processors.factory import FileLoaderFactory
import sys
from pathlib import Path
import time
import uuid
import tempfile
from typing import Dict, Any, List, Tuple
import networkx as nx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class KnowledgeGraphService:
    """Service for processing files to knowledge graph pipeline."""

    def __init__(
        self,
        extraction_chain: ExtractionChain,
        embedding_client: EmbeddingClient,
        chroma_store: ConceptChromaStore
    ):
        """
        Initialize service.

        Args:
            extraction_chain: ExtractionChain instance
            embedding_client: EmbeddingClient instance
            chroma_store: ConceptChromaStore instance
        """
        self.extraction_chain = extraction_chain
        self.embedding_client = embedding_client
        self.chroma_store = chroma_store

    async def process_document(
        self,
        file_path: str,
        subject_id: str,
        job_id: str,
        # Processing parameters
        num_chunks: int = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        batch_size: int = 5,
        max_workers: int = 8,
        # Merging parameters
        enable_name_merge: bool = True,
        enable_embedding_merge: bool = True,
        similarity_threshold: float = 0.9,
        # Prerequisite parameters
        prs_threshold: float = 0.6,
        # Verification parameters
        min_confidence: float = 0.5,
        # Graph parameters
        apply_reduction: bool = True,
        # Extraction context window
        max_previous_concepts: int = 20,
    ) -> ProcessingResult:
        """
        Process document (PDF, PPTX, etc.) through full knowledge graph pipeline.

        Args:
            file_path: Path to document file (PDF, PPTX, etc.)
            subject_id: Subject identifier
            job_id: Job identifier for tracking
            num_chunks: Number of chunks to process (None = all)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for extraction
            max_workers: Number of parallel workers
            enable_name_merge: Enable name-based merging
            enable_embedding_merge: Enable embedding-based deduplication
            similarity_threshold: Similarity threshold for deduplication
            prs_threshold: Threshold for prerequisite ranking
            min_confidence: Minimum confidence for verified relations
            apply_reduction: Apply transitive reduction

        Returns:
            ProcessingResult with complete statistics
        """
        start_time = time.time()
        result = ProcessingResult(
            job_id=job_id,
            subject_id=subject_id,
            status=ProcessingStatus.PROCESSING
        )

        try:
            # Step 1: Load & Chunk Document using Factory (auto-detects file type)
            chunks = FileLoaderFactory.load_and_chunk(
                file_path=file_path,
                doc_id=subject_id
            )

            # Limit chunks if specified
            if num_chunks:
                chunks = chunks[:num_chunks]

            result.total_chunks = len(chunks)

            # Step 2: Extract Concepts
            chunk_texts = [chunk.page_content for chunk in chunks]

            extractions = self.extraction_chain.extract_from_batch(
                chunks=chunk_texts,
                subject_id=subject_id,
                batch_size=batch_size,
                max_workers=max_workers,
                max_previous_concepts=max_previous_concepts,
            )

            all_concepts = []
            for extraction in extractions:
                all_concepts.extend(extraction.concepts)

            result.concepts_extracted = len(all_concepts)

            # Step 3: Generate Embeddings
            compute_embedding_for_concepts(
                concepts=all_concepts,
                client=self.embedding_client
            )

            # Step 4: Merge & Deduplicate
            concepts = all_concepts.copy()

            if enable_name_merge:
                concepts = merge_by_name(concepts)

            if enable_embedding_merge:
                concepts = deduplicate_by_embedding(
                    concepts,
                    similarity_threshold=similarity_threshold
                )

            result.concepts_after_merge = len(concepts)

            # Step 5: Find Prerequisite Pairs
            prereq_pairs = rank_prerequisites(
                concepts=concepts,
                prs_threshold=prs_threshold
            )

            # Step 6: Verify Relations
            qualifying_relations = []

            if prereq_pairs:
                concept_map = {c.concept_id: c for c in concepts}

                # Prepare pairs for verification
                concept_pairs = []
                pair_metadata = []

                for id1, id2 in prereq_pairs:
                    c1 = concept_map.get(id1)
                    c2 = concept_map.get(id2)

                    if c1 and c2:
                        concept_pairs.append((c1.name, c2.name))
                        pair_metadata.append({
                            'concept_a_id': id1,
                            'concept_b_id': id2,
                            'concept_a_name': c1.name,
                            'concept_b_name': c2.name
                        })

                # Verify all pairs in parallel
                verifications = self.extraction_chain.verify_relations_batch(
                    concept_pairs=concept_pairs,
                    max_workers=max_workers
                )

                # Combine results
                verification_results = []
                for metadata, verification in zip(pair_metadata, verifications):
                    verification_results.append({
                        **metadata,
                        'verification': verification
                    })

                # Filter by confidence
                qualifying_relations = [
                    r for r in verification_results
                    if r['verification'].has_relation and r['verification'].confidence >= min_confidence
                ]

            result.relations_verified = len(qualifying_relations)

            # Step 7: Build Knowledge Graph
            builder = KnowledgeGraphBuilder(subject_id=subject_id)

            # Add verified relations to concepts
            concept_map = {c.concept_id: c for c in concepts}

            for rel_result in qualifying_relations:
                verification = rel_result['verification']
                concept_a_id = rel_result['concept_a_id']
                concept_b_id = rel_result['concept_b_id']

                # Determine direction and add relation
                if verification.direction == "A_to_B":
                    source_id = concept_a_id
                    target_id = concept_b_id
                elif verification.direction == "B_to_A":
                    source_id = concept_b_id
                    target_id = concept_a_id
                elif verification.direction == "bidirectional":
                    source_id = concept_a_id
                    target_id = concept_b_id
                else:
                    continue

                # Add relation to concept
                if source_id in concept_map:
                    source_concept = concept_map[source_id]
                    existing_targets = {
                        rel.target_id for rel in source_concept.relations}

                    if target_id not in existing_targets:
                        new_relation = Relation(
                            type="PREREQUISITE",
                            target_id=target_id,
                            confidence=verification.confidence,
                            evidence="\n".join(
                                verification.evidences) if verification.evidences else None
                        )
                        source_concept.relations.append(new_relation)

                    # Handle bidirectional
                    if verification.direction == "bidirectional" and target_id in concept_map:
                        target_concept = concept_map[target_id]
                        existing_targets = {
                            rel.target_id for rel in target_concept.relations}
                        if source_id not in existing_targets:
                            new_relation = Relation(
                                type="PREREQUISITE",
                                target_id=source_id,
                                confidence=verification.confidence,
                                evidence="\n".join(
                                    verification.evidences) if verification.evidences else None
                            )
                            target_concept.relations.append(new_relation)

            # Add all concepts to graph
            builder.add_concepts(list(concept_map.values()))

            # Get initial stats
            graph = builder.get_graph()
            initial_stats = builder.get_stats()

            # Step 8: Convert to DAG by removing cycles
            dag_stats = None
            if not nx.is_directed_acyclic_graph(graph):
                graph, dag_stats = make_dag_with_llm(
                    graph, llm=self.extraction_chain.llm)
                builder.graph = graph

            # Step 9: Apply transitive reduction if requested
            if apply_reduction:
                graph = apply_transitive_reduction(graph)
                builder.graph = graph

            # Get final stats
            final_stats = builder.get_stats()

            # Build graph stats response
            optimization_stats = {}
            if dag_stats:
                optimization_stats['dag_conversion'] = dag_stats

            if initial_stats.get('num_edges', 0) > final_stats['num_edges']:
                optimization_stats['transitive_reduction'] = {
                    'edges_removed': initial_stats['num_edges'] - final_stats['num_edges']
                }

            result.graph_stats = GraphStatsResponse(
                num_nodes=final_stats['num_nodes'],
                num_edges=final_stats['num_edges'],
                density=final_stats['density'],
                has_cycle=final_stats['has_cycle'],
                edge_types=final_stats.get('edge_types', {}),
                is_dag=not final_stats['has_cycle'],
                optimization_stats=optimization_stats if optimization_stats else None
            )

            # Step 10: Store concepts in ChromaDB
            added_ids = self.chroma_store.add_concepts(
                concepts=list(concept_map.values()),
                subject_id=subject_id
            )

            result.concepts_in_chromadb = len(added_ids)

            # Complete
            result.status = ProcessingStatus.COMPLETED
            result.processing_time_seconds = time.time() - start_time

            return result

        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            result.processing_time_seconds = time.time() - start_time
            raise
