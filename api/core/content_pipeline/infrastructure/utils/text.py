"""Text processing utilities."""

import re

from loguru import logger

try:
    from underthesea import text_normalize as _text_normalize
except ImportError:
    _text_normalize = None
    logger.warning(
        "underthesea is not installed; falling back to regex-only text normalization",
    )

def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    text = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹ\s.,!?()\"'-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    if _text_normalize is not None:
        text = _text_normalize(text)

    return text
