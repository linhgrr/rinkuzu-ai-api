"""Runtime helpers for the legacy content-processor integration."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from loguru import logger

from ....config import get_settings


CONTENT_PROCESSOR_SRC = str(
    Path(__file__).resolve().parents[4] / "content-processor" / "src"
)
if CONTENT_PROCESSOR_SRC not in sys.path:
    sys.path.insert(0, CONTENT_PROCESSOR_SRC)


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


def try_import_content_processor() -> tuple[bool, str | None]:
    try:
        from processors.factory import FileLoaderFactory  # noqa: F401
        from processors.chunkers.text_chunker import TextChunker  # noqa: F401
        from llm.extract_chain import ExtractionChain  # noqa: F401
        from llm.postprocess import postprocess_concepts  # noqa: F401
        from embed.embedding_client import EmbeddingClient  # noqa: F401
        from embed.embeddings import compute_embedding_for_concepts  # noqa: F401
        from embed.prereq_ranking import rank_prerequisites  # noqa: F401
        from merge.name_merge import merge_by_name  # noqa: F401
        from graph.builder import KnowledgeGraphBuilder  # noqa: F401
        from graph.cycle_removal import make_dag_with_llm  # noqa: F401
        from graph.reduction import apply_transitive_reduction  # noqa: F401
        return True, None
    except ImportError as exc:
        import traceback

        err = f"{exc}\n\nsys.path: {sys.path}\n\nTraceback:\n{traceback.format_exc()}"
        logger.warning(f"Content processor not available: {err}")
        return False, str(exc)


_cp_result = try_import_content_processor()
CONTENT_PROCESSOR_AVAILABLE = _cp_result[0]
CONTENT_PROCESSOR_ERROR = _cp_result[1]
