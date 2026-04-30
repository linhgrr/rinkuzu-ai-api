"""Text chunker using LangChain built-ins (Recursive/Markdown/HF tokenizer)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None  # type: ignore[assignment,misc]
    _HAS_TRANSFORMERS = False

from loguru import logger

from api.config import settings


class TextChunker:
    """
    Chunker cho text/PDF dựa vào LangChain.
    - Bước 1: Nếu văn bản có header kiểu Markdown/đề mục -> tách theo header (giữ metadata header).
    - Bước 2: Re-chunk về kích thước mục tiêu bằng RecursiveCharacterTextSplitter (đếm theo ký tự).
    - Nếu dùng HuggingFace Embedding và muốn đếm theo token: bật use_hf_tokenizer=True.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *,
        use_hf_tokenizer: bool = False,
        hf_model_name: str | None = None,
        markdown_headers: list[tuple[str, str]] | None = None,
    ):
        """
        Args:
            chunk_size: Kích thước chunk (ký tự hoặc token, tùy splitter).
            chunk_overlap: Overlap giữa các chunk.
            use_hf_tokenizer: True nếu muốn đếm theo token với HF tokenizer
                              (chỉ nên dùng khi bạn chọn HuggingFace Embedding).
            hf_model_name: Tên tokenizer HF (ví dụ: "sentence-transformers/all-MiniLM-L6-v2")
                           bỏ trống sẽ dùng settings.embedding_model nếu có.
            markdown_headers: Danh sách header để tách markdown, mặc định theo #/##/###.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.use_hf_tokenizer = use_hf_tokenizer
        self.hf_model_name = hf_model_name or settings.embedding_model
        self.markdown_headers = markdown_headers or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        if self.use_hf_tokenizer and not _HAS_TRANSFORMERS:
            logger.warning(
                "use_hf_tokenizer=True nhưng thiếu 'transformers'. "
                "Sẽ fallback về đếm theo ký tự."
            )
            self.use_hf_tokenizer = False

    def chunk(self, content: dict[str, Any], doc_id: str) -> list[Document]:
        """
        Args:
            content: dict từ loader (yêu cầu có 'text', tùy chọn 'pages' và 'metadata')
            doc_id: ID tài liệu

        Returns:
            List[Document] (page_content + metadata)
        """
        text = (content or {}).get("text", "") or ""
        pages = (content or {}).get("pages", []) or []
        base_meta = dict((content or {}).get("metadata", {}))
        base_meta.update({"doc_id": doc_id})

        if not text.strip():
            logger.warning(f"No text to chunk for doc_id={doc_id}")
            return []

        # 1) Nếu nhìn giống có đề mục -> tách theo header markdown
        has_headers = self._looks_like_markdown_or_headings(text)
        docs: list[Document]
        if has_headers:
            md_splitter = MarkdownHeaderTextSplitter(self.markdown_headers)
            # -> List[Document] với metadata header
            header_docs = md_splitter.split_text(text)
            # Gắn metadata chung
            docs = [
                Document(page_content=d.page_content,
                         metadata={**base_meta, **d.metadata})
                for d in header_docs
            ]
        else:
            # Không có header: làm 1 doc gốc
            docs = [Document(page_content=text, metadata=base_meta)]

        # 2) Re-chunk về kích thước mục tiêu
        splitter = self._build_text_splitter()
        final_docs = splitter.split_documents(docs)

        # 3) Thêm ước lượng page range (nếu cần)
        # Nếu loader đã có mapping trang -> bạn có thể gắn vào metadata ở đây.
        for i, d in enumerate(final_docs):
            d.metadata.setdefault("chunk_index", i)
            # Ước lượng đơn giản: từ 1..len(pages) nếu có, else 1..1
            d.metadata.setdefault("start_page", 1)
            d.metadata.setdefault("end_page", max(1, len(pages)))

        logger.info(f"Created {len(final_docs)} chunks from document {doc_id}")
        return final_docs

    # -----------------------
    # Helpers
    # -----------------------

    def _build_text_splitter(self) -> TextSplitter:
        """
        Tạo TextSplitter cho bước re-chunk:
        - Nếu use_hf_tokenizer=True và có HF tokenizer: dùng from_huggingface_tokenizer (đếm theo token).
        - Ngược lại dùng RecursiveCharacterTextSplitter (đếm theo ký tự).
        """
        if self.use_hf_tokenizer and self.hf_model_name and AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_model_name, trust_remote_code=True)
                splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            except Exception as e:
                logger.warning(
                    f"HF tokenizer init failed ({e}). Fallback RecursiveCharacterTextSplitter."
                )
            else:
                logger.debug("Using HF tokenizer-based TextSplitter")
                return splitter

        # Fallback / mặc định: RecursiveCharacterTextSplitter (ổn định, khuyến nghị). :contentReference[oaicite:5]{index=5}
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # separators ưu tiên xuống dòng/space để hạn chế vỡ câu tiếng Việt
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
        )

    def _looks_like_markdown_or_headings(self, text: str) -> bool:
        """
        Heuristic nhận diện tài liệu có đề mục để ưu tiên MarkdownHeader splitter.
        """
        if re.search(r"^#{1,6}\s+\S+", text, flags=re.MULTILINE):
            return True  # Markdown style
        # Dạng "1. Giới thiệu", "Chương 1", "Mục 2", "Section 2" ...
        return bool(re.search(r"(?m)^(?:\d+\.\s+|Chương|Mục|Phần|Chapter|Section)\b", text, flags=re.IGNORECASE))
