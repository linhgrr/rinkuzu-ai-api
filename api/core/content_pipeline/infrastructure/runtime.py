"""Runtime helpers for the unified content pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import boto3
from botocore.client import Config
from loguru import logger

from ....config import get_settings
from ..application.ports import RelationEngine


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTENT_PROCESSOR_SRC = str(PROJECT_ROOT)


@dataclass(frozen=True)
class ContentProcessorBindings:
    """Imported collaborators used by the unified content pipeline."""

    file_loader_factory: Any
    extraction_chain_cls: Any
    postprocess_concepts: Callable[[list[Any]], list[Any]]
    llm_factory: Callable[..., Any]
    embedding_client_cls: Any
    compute_embedding_for_concepts: Callable[[list[Any], Any], Any]
    merge_by_name: Callable[[list[Any]], list[Any]]
    relation_engine_factory: Callable[..., RelationEngine]
    knowledge_graph_builder_factory: Callable[[str], Any]
    make_dag_with_llm: Callable[[Any], tuple[Any, Any]]
    apply_transitive_reduction: Callable[[Any], Any]
    saint_text_model_factory: Callable[[], Any]
    generate_theory: Callable[[str, str], Any]


def _generate_theory_via_exercise_gen(concept_name: str, concept_definition: str):
    """Import theory generation lazily to avoid runtime circular imports."""
    from ...exercise_gen import generate_theory

    return generate_theory(concept_name, concept_definition)


def get_s3_client():
    settings = get_settings()
    if not settings.s3_available:
        return None

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=Config(s3={"addressing_style": "path"}),
    )


def calculate_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        while chunk := file_obj.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_content_processor_bindings() -> ContentProcessorBindings:
    from ..application.relation_engine import DefaultRelationEngine
    from .processors.factory import FileLoaderFactory
    from .llm.extract_chain import ExtractionChain
    from .llm.postprocess import postprocess_concepts
    from .llm import get_llm
    from .embed.embedding_client import EmbeddingClient
    from .embed.embeddings import compute_embedding_for_concepts
    from .embed.prereq_ranking import rank_prerequisites
    from .merge.name_merge import merge_by_name
    from .graph.builder import KnowledgeGraphBuilder
    from .graph.cycle_removal import make_dag_with_llm
    from .graph.reduction import apply_transitive_reduction
    from sentence_transformers import SentenceTransformer

    return ContentProcessorBindings(
        file_loader_factory=FileLoaderFactory,
        extraction_chain_cls=ExtractionChain,
        postprocess_concepts=postprocess_concepts,
        llm_factory=get_llm,
        embedding_client_cls=EmbeddingClient,
        compute_embedding_for_concepts=compute_embedding_for_concepts,
        merge_by_name=merge_by_name,
        relation_engine_factory=lambda *, extraction_chain: DefaultRelationEngine(
            rank_prerequisites=rank_prerequisites,
            verify_relations_batch=extraction_chain.verify_relations_batch,
        ),
        knowledge_graph_builder_factory=lambda subject_id: KnowledgeGraphBuilder(subject_id=subject_id),
        make_dag_with_llm=make_dag_with_llm,
        apply_transitive_reduction=apply_transitive_reduction,
        saint_text_model_factory=lambda: SentenceTransformer("paraphrase-multilingual-mpnet-base-v2"),
        generate_theory=_generate_theory_via_exercise_gen,
    )


_content_processor_bindings: ContentProcessorBindings | None = None
_content_processor_llm_factory = None


def get_content_processor_bindings() -> ContentProcessorBindings:
    """Load and cache the imported collaborators for the content pipeline."""
    global _content_processor_bindings
    if _content_processor_bindings is None:
        _content_processor_bindings = _build_content_processor_bindings()
    return _content_processor_bindings


def get_content_processor_llm_factory():
    """Load and cache the content pipeline LLM factory."""
    global _content_processor_llm_factory
    if _content_processor_llm_factory is None:
        from .llm import get_llm

        _content_processor_llm_factory = get_llm
    return _content_processor_llm_factory


def try_import_content_processor() -> tuple[bool, str | None]:
    try:
        _build_content_processor_bindings()
        return True, None
    except Exception as exc:
        logger.warning(f"Content pipeline modules not available: {exc}")
        return False, str(exc)


_cp_result = try_import_content_processor()
CONTENT_PROCESSOR_AVAILABLE = _cp_result[0]
CONTENT_PROCESSOR_ERROR = _cp_result[1]
