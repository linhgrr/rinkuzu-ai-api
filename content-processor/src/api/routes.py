"""API routes for knowledge graph processing."""

import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.chroma_store import ConceptChromaStore
from embed.embedding_client import EmbeddingClient
from llm.extract_chain import ExtractionChain
from api.services import KnowledgeGraphService
from api.dependencies import (
    get_extraction_chain,
    get_embedding_client,
    get_chroma_store
)
from api.config import api_settings
from api.models import (
    UploadResponse,
    ProcessingResult,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ErrorResponse
)

import os
import uuid
import aiofiles
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks



router = APIRouter(prefix="/api/v1", tags=["Knowledge Graph"])

# In-memory job storage (in production, use Redis or database)
_jobs: dict = {}


def cleanup_file(file_path: str):
    """Background task to cleanup uploaded file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload document and start processing",
    description="Upload a document file (PDF, etc.) to be processed into a knowledge graph"
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...,
                            description="Document file to process (PDF, etc.)"),
    subject_id: str = Form(..., description="Subject identifier"),
    num_chunks: Optional[int] = Form(
        None, description="Number of chunks to process (None = all)"),
    chunk_size: int = Form(api_settings.default_chunk_size,
                           description="Size of text chunks"),
    chunk_overlap: int = Form(
        api_settings.default_chunk_overlap, description="Overlap between chunks"),
    batch_size: int = Form(api_settings.default_batch_size,
                           description="Batch size for extraction"),
    max_workers: int = Form(api_settings.default_max_workers,
                            description="Number of parallel workers"),
    enable_name_merge: bool = Form(
        True, description="Enable name-based merging"),
    enable_embedding_merge: bool = Form(
        True, description="Enable embedding-based deduplication"),
    similarity_threshold: float = Form(
        api_settings.default_similarity_threshold, description="Similarity threshold"),
    prs_threshold: float = Form(
        api_settings.default_prs_threshold, description="PRS threshold"),
    min_confidence: float = Form(
        api_settings.default_min_confidence, description="Minimum confidence"),
    apply_reduction: bool = Form(
        True, description="Apply transitive reduction"),
    max_previous_concepts: int = Form(
        api_settings.default_max_previous_concepts,
        description="Max number of previously extracted concept names to include in each batch prompt (window size)"),
    extraction_chain: ExtractionChain = Depends(get_extraction_chain),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    chroma_store: ConceptChromaStore = Depends(get_chroma_store)
):
    """Upload document file and start knowledge graph processing."""

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # Check file size
    file_size = 0
    chunk_data = await file.read()
    file_size = len(chunk_data)

    if file_size > api_settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {api_settings.max_upload_size / 1024 / 1024}MB"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save file
    upload_dir = Path(api_settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{job_id}_{file.filename}"

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(chunk_data)

    # Initialize service
    service = KnowledgeGraphService(
        extraction_chain=extraction_chain,
        embedding_client=embedding_client,
        chroma_store=chroma_store
    )

    # Store job info
    _jobs[job_id] = {
        "status": "pending",
        "subject_id": subject_id,
        "filename": file.filename
    }

    # Start background processing
    async def process_job():
        try:
            result = await service.process_document(
                file_path=str(file_path),
                subject_id=subject_id,
                job_id=job_id,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                max_workers=max_workers,
                enable_name_merge=enable_name_merge,
                enable_embedding_merge=enable_embedding_merge,
                similarity_threshold=similarity_threshold,
                prs_threshold=prs_threshold,
                min_confidence=min_confidence,
                apply_reduction=apply_reduction,
                max_previous_concepts=max_previous_concepts,
            )
            _jobs[job_id] = result.model_dump()
        except Exception as e:
            _jobs[job_id] = {
                "status": "failed",
                "error_message": str(e),
                "job_id": job_id,
                "subject_id": subject_id
            }
        finally:
            # Cleanup file after processing
            cleanup_file(str(file_path))

    background_tasks.add_task(process_job)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        file_size=file_size,
        subject_id=subject_id,
        message="File uploaded successfully. Processing started."
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ProcessingResult,
    summary="Get job status",
    description="Retrieve the status and results of a processing job"
)
async def get_job_status(job_id: str):
    """Get status of a processing job."""

    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    job_data = _jobs[job_id]

    # Convert to ProcessingResult if needed
    if isinstance(job_data, dict):
        return ProcessingResult(**job_data)

    return job_data


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search concepts",
    description="Search for concepts using semantic search"
)
async def search_concepts(
    request: SearchRequest,
    chroma_store: ConceptChromaStore = Depends(get_chroma_store)
):
    """Search for concepts in ChromaDB."""

    try:
        results = chroma_store.search_concepts(
            query=request.query,
            subject_id=request.subject_id,
            k=request.k
        )

        search_results = [
            SearchResult(**result) for result in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/concepts/{concept_id}",
    summary="Get concept by ID",
    description="Retrieve a specific concept by its ID"
)
async def get_concept(
    concept_id: str,
    chroma_store: ConceptChromaStore = Depends(get_chroma_store)
):
    """Get a concept by ID."""

    try:
        concept = chroma_store.get_concept_by_id(concept_id)

        if concept is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )

        return concept

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve concept: {str(e)}"
        )


@router.delete(
    "/subjects/{subject_id}",
    summary="Delete subject",
    description="Delete all concepts for a subject"
)
async def delete_subject(
    subject_id: str,
    chroma_store: ConceptChromaStore = Depends(get_chroma_store)
):
    """Delete all concepts for a subject."""

    try:
        deleted_count = chroma_store.delete_by_subject(subject_id)

        return {
            "message": f"Deleted {deleted_count} concepts for subject {subject_id}",
            "subject_id": subject_id,
            "deleted_count": deleted_count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete subject: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get ChromaDB statistics",
    description="Get statistics about the ChromaDB collection"
)
async def get_stats(
    chroma_store: ConceptChromaStore = Depends(get_chroma_store)
):
    """Get ChromaDB collection statistics."""

    try:
        stats = chroma_store.get_collection_stats()
        return stats

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
