"""Tests for _verify_single_relation retry behaviour.

The manual for-loop retry in _verify_single_relation must:
  1. Retry on transient failures and ultimately return the result.
  2. Return a _verification_error sentinel (NOT raise) after exhausting all attempts.
"""

from __future__ import annotations

import asyncio

from api.domains.content_pipeline.infrastructure.llm.extract_chain import ExtractionChain
from api.domains.content_pipeline.infrastructure.llm.schemas import EvidenceVerification

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _SucceedAfterNClient:
    """Fails n times then returns a valid EvidenceVerification."""

    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self.fail_times = fail_times

    async def parse_response(self, **_: object) -> EvidenceVerification:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError(f"transient failure #{self.calls}")
        return EvidenceVerification(
            has_relation=True,
            relation_type="PREREQUISITE",
            direction="A_to_B",
            confidence=0.9,
            evidences=["some evidence"],
            reasoning="succeeded after retries",
        )


class _AlwaysFailClient:
    """Always raises — used to test exhaustion path."""

    def __init__(self) -> None:
        self.calls = 0

    async def parse_response(self, **_: object) -> EvidenceVerification:
        self.calls += 1
        raise RuntimeError("provider permanently down")


class _NoopDocumentExtractor:
    """Satisfies ExtractionChain constructor without real I/O."""

    def extract_file(self, _file_path: str) -> object:
        from api.shared.document_text import ExtractedDocumentText

        return ExtractedDocumentText(text="", pages=[], metadata={"page_count": 0})


# ---------------------------------------------------------------------------
# Test: succeeds after transient failures
# ---------------------------------------------------------------------------


def test_verify_single_relation_returns_result_on_success():
    """A successful parse flows straight through to the verified relation.

    Transient-failure retry now lives in the LLM client (below parse_response);
    see test_llm_retry_in_client for the attempt-count contract. This fake
    client sits above the retry point, so it models a single already-resolved
    call.
    """
    client = _SucceedAfterNClient(fail_times=0)
    chain = ExtractionChain(client=client, document_extractor=_NoopDocumentExtractor())

    result = asyncio.run(
        chain._verify_single_relation(
            concept_a="Định luật Ohm",
            concept_b="Điện trở",
            pair_idx=0,
        )
    )

    assert result.has_relation is True
    assert result.relation_type == "PREREQUISITE"
    assert client.calls == 1


# ---------------------------------------------------------------------------
# Test: exhaustion → _verification_error sentinel (does NOT raise)
# ---------------------------------------------------------------------------


def test_verify_single_relation_returns_error_sentinel_on_exhaustion(monkeypatch):
    """After all retries are used up, must return a _verification_error, not raise."""
    import api.shared.retry as retry_module

    # Only 2 attempts so the always-failing client exhausts quickly.
    monkeypatch.setattr(retry_module, "resolve_llm_retry_policy", lambda: (2, 0.0))

    client = _AlwaysFailClient()
    chain = ExtractionChain(client=client, document_extractor=_NoopDocumentExtractor())

    result = asyncio.run(
        chain._verify_single_relation(
            concept_a="A",
            concept_b="B",
            pair_idx=1,
            max_retries=2,
        )
    )

    # Must NOT raise — must return the sentinel
    assert isinstance(result, EvidenceVerification)
    assert result.has_relation is False
    assert result.confidence == 0.0
    assert result.reasoning is not None
    assert (
        "Failed" in result.reasoning
        or "failed" in result.reasoning
        or "unavailable" in result.reasoning
    )
