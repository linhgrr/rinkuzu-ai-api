import asyncio

import fitz
from pydantic import ValidationError
import pytest

from api.core.content_pipeline.infrastructure.llm import extract_chain as extract_chain_module
from api.core.content_pipeline.infrastructure.llm.extract_chain import (
    ExtractionChain,
    ProviderUploadTooLargeError,
    build_page_batches,
)
from api.core.content_pipeline.infrastructure.llm.schemas import (
    ConceptExtraction,
    ConceptExtractionPayload,
    materialize_concept_extraction,
)
from api.core.content_pipeline.infrastructure.processors.loaders.local_pdf_text_loader import (
    load_pdf,
)


class _ParsedResponse:
    def __init__(self, parsed: object | None, *, usage: dict[str, int] | None = None, output_text: str = ""):
        self.output_parsed = parsed
        self.usage = usage or {}
        self.output_text = output_text


class _RetryClient:
    def __init__(self):
        self.calls = 0

    async def parse_response(self, **_: object) -> _ParsedResponse:
        self.calls += 1
        if self.calls == 1:
            return _ParsedResponse(None, output_text="temporary malformed output")
        payload = ConceptExtractionPayload.model_validate(
            {
                "concepts": [],
                "subject_id": "math",
                "notes": None,
            }
        )
        return _ParsedResponse(payload, usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})


async def _noop_upload_pdf_bytes(*, filename, pdf_bytes, sha256, now_ts, job_id=None):
    from api.core.content_pipeline.infrastructure.llm.openai_responses import UploadedFileRef
    return UploadedFileRef(file_id="file-123", purpose="user_data", cache_hit=False)


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


def test_extraction_response_retries_on_invalid_structured_output():
    client = _RetryClient()
    chain = ExtractionChain(client=client)

    async def _run():
        payload, usage = await chain._invoke_extraction_response_with_retries(
            job_id="job-1",
            subject_id="math",
            file_id="file-123",
            previous_concepts=[],
            max_retries=2,
        )
        return payload, usage

    payload, usage = asyncio.run(_run())

    assert client.calls == 2
    assert payload.subject_id == "math"
    assert payload.concepts == []
    assert usage == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


def test_invoke_extraction_response_uses_pydantic_structured_output():
    captured: dict[str, object] = {}

    class _Client:
        async def parse_response(self, **kwargs: object) -> _ParsedResponse:
            captured.update(kwargs)
            payload = ConceptExtractionPayload.model_validate(
                {"concepts": [], "subject_id": "math", "notes": None}
            )
            return _ParsedResponse(payload)

    chain = ExtractionChain(client=_Client())  # type: ignore[arg-type]

    async def _run():
        payload, _usage = await chain._invoke_extraction_response(
            job_id="job-structured",
            subject_id="math",
            file_id="file-structured",
            previous_concepts=[("c1", "Khái niệm cũ")],
        )
        return payload

    payload = asyncio.run(_run())

    assert payload.subject_id == "math"
    assert captured["text_format"] is ConceptExtractionPayload
    input_blocks = captured["input_blocks"]
    assert isinstance(input_blocks, list)
    assert input_blocks[1]["type"] == "input_file"


def test_render_batched_pdfs_uses_compressed_pdf_when_it_fits(monkeypatch):
    chain = ExtractionChain(client=_RetryClient())

    def _fake_extract_pdf_bytes(_document, _start, _end):
        return b"a" * 10

    def _fake_extract_compressed_pdf_bytes(_document, _start, _end, *, dpi, jpg_quality):
        del dpi, jpg_quality
        return b"b" * 4

    monkeypatch.setattr(chain, "_extract_pdf_bytes", _fake_extract_pdf_bytes)
    monkeypatch.setattr(
        chain,
        "_extract_compressed_pdf_bytes",
        _fake_extract_compressed_pdf_bytes,
    )

    batches = chain._render_batched_pdfs(
        document=object(),
        batch_index=0,
        start_page=1,
        end_page=2,
        max_bytes=5,
    )

    assert len(batches) == 1
    assert batches[0]["size_bytes"] == 4
    assert batches[0]["compression_applied"] is True
    assert batches[0]["compression_profile"] == "144dpi-q75"


def test_local_pdf_text_loader_extracts_page_text(tmp_path):
    pdf_path = tmp_path / "lesson.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Xin chao the gioi")
    document.save(pdf_path)
    document.close()

    payload = load_pdf(str(pdf_path))

    assert payload["metadata"]["source"] == "pymupdf"
    assert payload["metadata"]["page_count"] == 1
    assert "Xin chao the gioi" in payload["text"]


def test_extract_from_document_splits_again_when_provider_rejects_upload_size(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan-like.pdf"
    document = fitz.open()
    for index in range(2):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1}")
    document.save(pdf_path)
    document.close()

    chain = ExtractionChain(client=_RetryClient())
    seen_ranges: list[tuple[int, int]] = []

    async def fake_extract_single_batch(
        *, job_id, subject_id, batch, previous_concepts, source_name
    ):
        del job_id, previous_concepts, source_name
        seen_ranges.append((batch["page_start"], batch["page_end"]))
        if (batch["page_start"], batch["page_end"]) == (1, 2):
            raise ProviderUploadTooLargeError("payload too large")
        return ConceptExtraction(concepts=[], subject_id=subject_id, notes=None)

    monkeypatch.setattr(chain, "_extract_single_batch", fake_extract_single_batch)

    async def fake_run_process_stage(target_path, *args, stage_name, timeout_sec=None, **kwargs):
        del stage_name, timeout_sec
        if target_path == extract_chain_module._READ_PAGE_COUNT_PROCESS_TARGET:
            return 2
        if target_path == extract_chain_module._RENDER_BATCHES_PROCESS_TARGET:
            return [
                {
                    "batch_index": kwargs["batch_index"],
                    "page_start": kwargs["start_page"],
                    "page_end": kwargs["end_page"],
                    "pdf_bytes": b"x",
                    "sha256": "abc",
                    "size_bytes": 1,
                }
            ]
        if target_path == extract_chain_module._SPLIT_BATCH_PROCESS_TARGET:
            _file_path = args[0]
            batch = kwargs["batch"]
            return [
                {
                    "batch_index": batch["batch_index"],
                    "page_start": 1,
                    "page_end": 1,
                    "pdf_bytes": b"x",
                    "sha256": "abc",
                    "size_bytes": 1,
                },
                {
                    "batch_index": batch["batch_index"],
                    "page_start": 2,
                    "page_end": 2,
                    "pdf_bytes": b"x",
                    "sha256": "abc",
                    "size_bytes": 1,
                },
            ]
        raise AssertionError(f"unexpected process target: {target_path}")

    monkeypatch.setattr(extract_chain_module, "run_process_stage", fake_run_process_stage)

    async def _run():
        return await chain.extract_from_document(str(pdf_path), "math", page_batch_size=10)

    results = asyncio.run(_run())

    assert len(results) == 2
    assert seen_ranges == [(1, 2), (1, 1), (2, 2)]
    assert [(batch["page_start"], batch["page_end"]) for batch in chain.last_batches] == [
        (1, 1),
        (2, 2),
    ]
    assert chain.last_failed_batches == []
