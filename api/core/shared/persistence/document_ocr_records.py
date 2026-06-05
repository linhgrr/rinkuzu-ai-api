from __future__ import annotations

from typing import Any

from loguru import logger

from .documents import DocumentOCRPage, DocumentOCRRecordDocument


def _normalize_pages(pages: list[dict[str, Any]] | None) -> list[DocumentOCRPage]:
    return [DocumentOCRPage.model_validate(page) for page in pages or []]


def _document_to_public_dict(doc: DocumentOCRRecordDocument) -> dict[str, Any]:
    return {
        "file_hash": doc.file_hash,
        "file_name": doc.file_name,
        "file_size_bytes": doc.file_size_bytes,
        "text": doc.text,
        "page_count": doc.page_count,
        "provider": doc.provider,
        "model": doc.model,
        "pages": [page.model_dump() for page in doc.pages],
        "metadata": dict(doc.metadata or {}),
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }


async def load_document_ocr_record(file_hash: str) -> dict[str, Any] | None:
    try:
        doc = await DocumentOCRRecordDocument.find_one(
            DocumentOCRRecordDocument.file_hash == file_hash
        )
    except Exception:
        logger.exception("[DocumentOCRStore] load failed file_hash={}", file_hash)
        return None
    return None if doc is None else _document_to_public_dict(doc)


async def save_document_ocr_record(
    *,
    file_hash: str,
    file_name: str,
    file_size_bytes: int | None,
    content: dict[str, Any],
) -> bool:
    try:
        metadata = dict(content.get("metadata") or {})
        pages = _normalize_pages(content.get("pages"))
        page_count = int(metadata.get("page_count") or len(pages))
        payload = {
            "file_hash": file_hash,
            "file_name": file_name,
            "file_size_bytes": file_size_bytes,
            "text": str(content.get("text") or ""),
            "page_count": page_count,
            "provider": metadata.get("provider"),
            "model": metadata.get("model"),
            "pages": pages,
            "metadata": metadata,
        }
        existing = await DocumentOCRRecordDocument.find_one(
            DocumentOCRRecordDocument.file_hash == file_hash
        )
        if existing is None:
            await DocumentOCRRecordDocument(**payload).insert()
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
            await existing.replace()
    except Exception:
        logger.exception("[DocumentOCRStore] save failed file_hash={}", file_hash)
        return False
    return True
