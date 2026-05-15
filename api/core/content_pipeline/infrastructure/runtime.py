"""Runtime helpers for the unified content pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
import hashlib
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import boto3
from botocore.client import Config
from loguru import logger

from api.config import get_settings
from api.core.content_pipeline.application.relation_engine import DefaultRelationEngine

from .graph.builder import KnowledgeGraphBuilder
from .graph.cycle_removal import make_dag_with_llm
from .graph.reduction import apply_transitive_reduction
from .llm.extract_chain import ExtractionChain
from .llm.postprocess import postprocess_concepts
from .merge.name_merge import merge_by_name
from .processors.factory import load_and_chunk_pdf

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from api.core.content_pipeline.application.ports import RelationEngine

PrereqRankingFn = Callable[[list[Any], float], list[tuple[str, str]]]

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTENT_PROCESSOR_SRC = str(PROJECT_ROOT)
_rank_prerequisites: PrereqRankingFn | None = None
_SentenceTransformerFactory: Callable[[str], Any] | None = None

try:
    from .embed.prereq_ranking import rank_prerequisites as _imported_rank_prerequisites

    _rank_prerequisites = _imported_rank_prerequisites
    _PREREQ_RANKING_AVAILABLE = True
except ModuleNotFoundError:
    _PREREQ_RANKING_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _ImportedSentenceTransformer

    _SentenceTransformerFactory = _ImportedSentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass(frozen=True)
class ContentProcessorBindings:
    """Imported collaborators used by the unified content pipeline."""

    file_loader_factory: Any
    extraction_chain_cls: Any
    postprocess_concepts: Callable[[list[Any]], list[Any]]
    embedding_client_factory: Callable[[str, int], Any]
    compute_embedding_for_concepts: Callable[[list[Any], Any], Any]
    merge_by_name: Callable[[list[Any]], list[Any]]
    relation_engine_factory: Callable[..., RelationEngine]
    knowledge_graph_builder_factory: Callable[[str], Any]
    make_dag_with_llm: Callable[[Any], Awaitable[tuple[Any, Any]]]
    apply_transitive_reduction: Callable[[Any], Any]
    saint_text_model_factory: Callable[[], Any]
    generate_theory: Callable[[str, str], Any]


class LockedSentenceTransformerModel:
    """Thread-safe wrapper for a shared sentence-transformers model."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._lock = Lock()

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode(*args, **kwargs)


def _generate_theory_via_exercise_gen(concept_name: str, concept_definition: str) -> Any:
    """Delegate theory generation to exercise_gen module."""
    exercise_gen = import_module("api.core.learning.exercise_gen")
    return exercise_gen.generate_theory(concept_name, concept_definition)


def get_s3_client() -> Any:
    settings = get_settings()
    if not settings.s3_available:
        return None

    return boto3.client(
        "s3",
        endpoint_url=settings.object_storage_client_endpoint,
        region_name=settings.object_storage_region,
        aws_access_key_id=settings.object_storage_access_key,
        aws_secret_access_key=settings.object_storage_secret_key,
        config=Config(s3={"addressing_style": settings.object_storage_addressing_style or "path"}),
    )


def calculate_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as file_obj:
        while chunk := file_obj.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_embedding_client(model_name: str, batch_size: int) -> Any:
    from .embed.embedding_client import EmbeddingClient

    return EmbeddingClient(model_name, batch_size=batch_size)


def _compute_embedding_for_concepts(concepts: Any, embedding_model: Any) -> Any:
    from .embed.embeddings import compute_embedding_for_concepts

    return compute_embedding_for_concepts(concepts, embedding_model)


@functools.lru_cache(maxsize=1)
def _load_saint_text_model() -> LockedSentenceTransformerModel:
    if not _SENTENCE_TRANSFORMERS_AVAILABLE or _SentenceTransformerFactory is None:
        raise ImportError("sentence-transformers is required for SAINT text model")
    model = _SentenceTransformerFactory("paraphrase-multilingual-mpnet-base-v2")
    return LockedSentenceTransformerModel(model)


def _build_relation_engine(*, extraction_chain: Any) -> Any:
    if not _PREREQ_RANKING_AVAILABLE:

        def rank_prerequisites_stub(
            _items: list[Any],
            _threshold: float,
        ) -> list[tuple[str, str]]:
            raise ModuleNotFoundError(
                "Optional embedding dependencies are required for prerequisite ranking"
            )

        rank_fn: PrereqRankingFn = rank_prerequisites_stub
    else:
        assert _rank_prerequisites is not None
        rank_fn = _rank_prerequisites

    return DefaultRelationEngine(
        rank_prerequisites=rank_fn,
        verify_relations_batch=extraction_chain.verify_relations_batch,
    )


def _merge_by_name(concepts: Any) -> Any:
    return merge_by_name(concepts)


def _knowledge_graph_builder_factory(subject_id: str) -> Any:
    return KnowledgeGraphBuilder(subject_id=subject_id)


async def _make_dag_with_llm(graph: Any) -> Any:
    return await make_dag_with_llm(graph)


def _apply_transitive_reduction(graph: Any) -> Any:
    return apply_transitive_reduction(graph)


def _build_saint_text_model() -> Any:
    return _load_saint_text_model()


def _build_content_processor_bindings() -> ContentProcessorBindings:
    return ContentProcessorBindings(
        file_loader_factory=load_and_chunk_pdf,
        extraction_chain_cls=ExtractionChain,
        postprocess_concepts=postprocess_concepts,
        embedding_client_factory=_build_embedding_client,
        compute_embedding_for_concepts=_compute_embedding_for_concepts,
        merge_by_name=_merge_by_name,
        relation_engine_factory=_build_relation_engine,
        knowledge_graph_builder_factory=_knowledge_graph_builder_factory,
        make_dag_with_llm=_make_dag_with_llm,
        apply_transitive_reduction=_apply_transitive_reduction,
        saint_text_model_factory=_build_saint_text_model,
        generate_theory=_generate_theory_via_exercise_gen,
    )


@functools.lru_cache(maxsize=1)
def get_content_processor_bindings() -> ContentProcessorBindings:
    """Load and cache the imported collaborators for the content pipeline."""
    return _build_content_processor_bindings()


def try_import_content_processor() -> tuple[bool, str | None]:
    try:
        _build_content_processor_bindings()
    except Exception as exc:
        logger.warning("Content pipeline modules not available: {}", exc)
        return False, str(exc)
    else:
        return True, None


_cp_result = try_import_content_processor()
CONTENT_PROCESSOR_AVAILABLE = _cp_result[0]
CONTENT_PROCESSOR_ERROR = _cp_result[1]
