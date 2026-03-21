"""Backward-compatible shim for legacy root imports."""

from api.core.content_pipeline.infrastructure.prompts import (  # noqa: F401
    CYCLE_REMOVAL_PROMPT,
    EVIDENCE_VERIFICATION_PROMPT,
    EXTRACTION_PROMPT,
    load_prompt,
)
