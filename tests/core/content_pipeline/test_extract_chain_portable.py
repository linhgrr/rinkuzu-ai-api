import asyncio

from pydantic import ValidationError
import pytest

from api.domains.content_pipeline.infrastructure.llm.extract_chain import ExtractionChain
from api.domains.content_pipeline.infrastructure.llm.schemas import (
    ConceptExtraction,
    ConceptExtractionPayload,
    materialize_concept_extraction,
)
from api.shared.document_text import (
    DocumentPageText,
    ExtractedDocumentText,
    build_page_batches,
    extract_document_text_from_file,
    extracted_document_text_to_content_payload,
)


class _RetryClient:
    def __init__(self):
        self.calls = 0

    async def parse_response(self, **_: object) -> ConceptExtractionPayload:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("temporary malformed output")
        return ConceptExtractionPayload.model_validate(
            {
                "concepts": [],
                "subject_id": "math",
                "notes": None,
            }
        )


class _NoopExtractor:
    def extract_file(self, _file_path: str) -> ExtractedDocumentText:
        return ExtractedDocumentText(text="", pages=[], metadata={"page_count": 0})


def test_build_page_batches_uses_fixed_windows():
    assert build_page_batches(23, 10) == [(1, 10), (11, 20), (21, 23)]


def test_materialize_concept_extraction_sets_optional_embeddings_to_none():
    payload = ConceptExtractionPayload.model_validate(
        {
            "concepts": [
                {
                    "concept_id": "dinh_luat_ohm",
                    "subject_id": "physics",
                    "name": "Định luật Ohm",
                    "definition": "Mối quan hệ giữa cường độ dòng điện, hiệu điện thế và điện trở.",
                    "examples": ["I = U / R"],
                    "formulas": [],
                    "relations": [],
                }
            ],
            "subject_id": "physics",
            "notes": None,
        }
    )

    extraction = materialize_concept_extraction(payload)

    assert extraction.concepts[0].name_embedding is None
    assert extraction.concepts[0].definition_embedding is None


def test_payload_models_forbid_unknown_fields():
    with pytest.raises(ValidationError, match="unexpected"):
        ConceptExtractionPayload.model_validate(
            {
                "concepts": [],
                "subject_id": "physics",
                "notes": None,
                "unexpected": "boom",
            }
        )


def test_extraction_response_propagates_client_error():
    """_invoke_extraction_response no longer retries — transient-failure retry
    moved into the LLM client (below parse_response). A client error here
    propagates; the batch-level caller turns it into an error payload (see
    test_extract_single_batch_returns_error_payload_when_client_fails).
    """
    client = _RetryClient()
    chain = ExtractionChain(client=client, document_extractor=_NoopExtractor())

    async def _run():
        return await chain._invoke_extraction_response(
            job_id="job-1",
            subject_id="math",
            document_text="Đây là nội dung cần trích xuất.",
            previous_concepts=[],
        )

    with pytest.raises(ValueError, match="temporary malformed output"):
        asyncio.run(_run())
    assert client.calls == 1


def test_invoke_extraction_response_uses_structured_generation_client():
    captured: dict[str, object] = {}

    class _Client:
        async def parse_response(self, **kwargs: object) -> ConceptExtractionPayload:
            captured.update(kwargs)
            return ConceptExtractionPayload.model_validate(
                {"concepts": [], "subject_id": "math", "notes": None}
            )

    chain = ExtractionChain(  # type: ignore[arg-type]
        client=_Client(),
        document_extractor=_NoopExtractor(),
    )

    async def _run():
        return await chain._invoke_extraction_response(
            job_id="job-structured",
            subject_id="math",
            document_text="## Trang 1\nKhái niệm cũ nối sang nội dung mới.",
            previous_concepts=[("c1", "Khái niệm cũ")],
        )

    payload = asyncio.run(_run())

    assert payload.subject_id == "math"
    assert captured["text_format"] is ConceptExtractionPayload
    assert "Khái niệm cũ" in str(captured["user_text"])
    assert "<document_text>" in str(captured["user_text"])


def test_extract_single_batch_returns_error_payload_when_client_fails():
    chain = ExtractionChain(client=_RetryClient(), document_extractor=_NoopExtractor())

    async def fake_invoke(**_kwargs):
        raise RuntimeError("provider down")

    chain._invoke_extraction_response = fake_invoke  # type: ignore[method-assign]

    async def _run():
        return await chain._extract_single_batch(
            job_id="job-1",
            subject_id="math",
            batch={
                "batch_index": 0,
                "page_start": 1,
                "page_end": 2,
                "text": "## Trang 1\nNội dung",
                "char_count": 20,
            },
            previous_concepts=[],
            source_name="source.pdf",
        )

    result = asyncio.run(_run())

    assert isinstance(result, ConceptExtraction)
    assert result.notes is not None
    assert result.notes.startswith("Error:")


def test_extract_document_text_from_file_can_be_reshaped_for_chunking(tmp_path):
    pdf_path = tmp_path / "lesson.pdf"
    pdf_path.write_bytes(b"%PDF-demo")

    class _Extractor:
        def extract_file(self, _file_path: str) -> ExtractedDocumentText:
            return ExtractedDocumentText(
                text="## Trang 1\nXin chao the gioi",
                pages=[DocumentPageText(page_number=1, text="Xin chao the gioi")],
                metadata={"source": "ocr_api", "page_count": 1},
            )

    def _build_extractor():
        return _Extractor()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "api.shared.document_text.build_document_text_extractor",
        _build_extractor,
    )

    try:
        payload = extracted_document_text_to_content_payload(
            extract_document_text_from_file(str(pdf_path))
        )
    finally:
        monkeypatch.undo()

    assert payload["metadata"]["source"] == "ocr_api"
    assert payload["metadata"]["page_count"] == 1
    assert "Xin chao the gioi" in payload["text"]


def test_extract_from_document_uses_text_batches_and_tracks_failed_batches():
    document_text = ExtractedDocumentText(
        text="## Trang 1\nAlpha\n\n## Trang 2\nBeta",
        pages=[
            DocumentPageText(page_number=1, text="Alpha"),
            DocumentPageText(page_number=2, text="Beta"),
        ],
        metadata={"page_count": 2, "file_name": "sample.pdf"},
    )

    class _Extractor:
        def extract_file(self, _file_path: str) -> ExtractedDocumentText:
            return document_text

    chain = ExtractionChain(client=_RetryClient(), document_extractor=_Extractor())
    seen_ranges: list[tuple[int, int]] = []

    async def fake_extract_single_batch(
        *, job_id, subject_id, batch, previous_concepts, source_name
    ):
        del job_id, previous_concepts, source_name
        seen_ranges.append((batch["page_start"], batch["page_end"]))
        if batch["page_start"] == 2:
            return ConceptExtraction(
                concepts=[], subject_id=subject_id, notes="Error: missing signal"
            )
        return ConceptExtraction(concepts=[], subject_id=subject_id, notes=None)

    chain._extract_single_batch = fake_extract_single_batch  # type: ignore[method-assign]

    async def _run():
        return await chain.extract_from_document("sample.pdf", "math", page_batch_size=1)

    results = asyncio.run(_run())

    assert len(results) == 2
    assert seen_ranges == [(1, 1), (2, 2)]
    assert [(batch["page_start"], batch["page_end"]) for batch in chain.last_batches] == [
        (1, 1),
        (2, 2),
    ]
    assert chain.last_failed_batches == [
        {
            "batch_index": 1,
            "page_start": 2,
            "page_end": 2,
            "reason": "Error: missing signal",
        }
    ]


def test_extract_from_document_fires_on_batch_progress_once_per_batch():
    document_text = ExtractedDocumentText(
        text="## Trang 1\nAlpha\n\n## Trang 2\nBeta\n\n## Trang 3\nGamma",
        pages=[
            DocumentPageText(page_number=1, text="Alpha"),
            DocumentPageText(page_number=2, text="Beta"),
            DocumentPageText(page_number=3, text="Gamma"),
        ],
        metadata={"page_count": 3, "file_name": "sample.pdf"},
    )

    class _Extractor:
        def extract_file(self, _file_path: str) -> ExtractedDocumentText:
            return document_text

    chain = ExtractionChain(client=_RetryClient(), document_extractor=_Extractor())

    async def fake_extract_single_batch(
        *, job_id, subject_id, batch, previous_concepts, source_name
    ):
        del job_id, previous_concepts, source_name
        return ConceptExtraction(concepts=[], subject_id=subject_id, notes=None)

    chain._extract_single_batch = fake_extract_single_batch  # type: ignore[method-assign]

    progress_calls: list[tuple[int, int]] = []

    async def on_batch_progress(done: int, total: int) -> None:
        progress_calls.append((done, total))

    async def _run():
        return await chain.extract_from_document(
            "sample.pdf",
            "math",
            page_batch_size=1,
            on_batch_progress=on_batch_progress,
        )

    results = asyncio.run(_run())

    assert len(results) == 3
    # One heartbeat per batch, reporting the running (completed, total) counts.
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]


def test_extract_from_document_survives_failing_on_batch_progress():
    document_text = ExtractedDocumentText(
        text="## Trang 1\nAlpha\n\n## Trang 2\nBeta",
        pages=[
            DocumentPageText(page_number=1, text="Alpha"),
            DocumentPageText(page_number=2, text="Beta"),
        ],
        metadata={"page_count": 2, "file_name": "sample.pdf"},
    )

    class _Extractor:
        def extract_file(self, _file_path: str) -> ExtractedDocumentText:
            return document_text

    chain = ExtractionChain(client=_RetryClient(), document_extractor=_Extractor())

    async def fake_extract_single_batch(
        *, job_id, subject_id, batch, previous_concepts, source_name
    ):
        del job_id, previous_concepts, source_name
        return ConceptExtraction(concepts=[], subject_id=subject_id, notes=None)

    chain._extract_single_batch = fake_extract_single_batch  # type: ignore[method-assign]

    attempts: list[int] = []

    async def exploding_on_batch_progress(done: int, total: int) -> None:
        del total
        attempts.append(done)
        raise RuntimeError("mongo heartbeat blip")

    async def _run():
        return await chain.extract_from_document(
            "sample.pdf",
            "math",
            page_batch_size=1,
            on_batch_progress=exploding_on_batch_progress,
        )

    # A failing heartbeat must NOT abort extraction.
    results = asyncio.run(_run())

    assert len(results) == 2
    # Callback was still attempted for every batch despite raising.
    assert attempts == [1, 2]


def test_extract_from_document_reuses_provided_document_text_without_ocr():
    document_text = ExtractedDocumentText(
        text="## Trang 1\nAlpha",
        pages=[DocumentPageText(page_number=1, text="Alpha")],
        metadata={"page_count": 1, "file_name": "sample.pdf"},
    )

    class _Extractor:
        def extract_file(self, _file_path: str) -> ExtractedDocumentText:
            raise AssertionError("should reuse pipeline OCR text")

    chain = ExtractionChain(client=_RetryClient(), document_extractor=_Extractor())
    seen_ranges: list[tuple[int, int]] = []

    async def fake_extract_single_batch(
        *, job_id, subject_id, batch, previous_concepts, source_name
    ):
        del job_id, previous_concepts, source_name
        seen_ranges.append((batch["page_start"], batch["page_end"]))
        return ConceptExtraction(concepts=[], subject_id=subject_id, notes=None)

    chain._extract_single_batch = fake_extract_single_batch  # type: ignore[method-assign]

    async def _run():
        return await chain.extract_from_document(
            "sample.pdf",
            "math",
            page_batch_size=1,
            document_text=document_text,
        )

    results = asyncio.run(_run())

    assert len(results) == 1
    assert seen_ranges == [(1, 1)]
