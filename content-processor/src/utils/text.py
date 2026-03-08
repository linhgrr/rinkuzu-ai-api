"""Text processing utilities."""

import re
from underthesea import text_normalize

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
    
    text = text_normalize(text)
    
    return text
