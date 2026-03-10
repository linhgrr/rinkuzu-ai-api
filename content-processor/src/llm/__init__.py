import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _normalize_openai_base_url(url: str) -> str:
    """Ensure OpenAI-compatible base URL includes /v1 prefix path."""
    raw = (url or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def get_llm(temperature=0.0, **kwargs):
    """
    Returns a configured ChatOpenAI instance pointing to a local compatible API.
    """
    base_url_raw = kwargs.pop("base_url", None)
    if not base_url_raw:
        base_url_raw = os.getenv("LLM_BASE_URL", "http://localhost:6969")
    base_url = _normalize_openai_base_url(base_url_raw)
    
    model = kwargs.pop("model", None) or os.getenv("LLM_MODEL", "gemini-3.0-pro")
    api_key = (
        kwargs.pop("api_key", None)
        or os.getenv("LLM_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or "sk-41bb5a29c07d4b23ad5e8e54a658ce2b"
    )
    
    # Remove kwargs that are specific to ChatGoogleGenerativeAI
    kwargs.pop("max_retries", None)

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
        timeout=150,
        **kwargs
    )

def get_embeddings(**kwargs):
    """
    Returns a configured OpenAIEmbeddings instance pointing to a local compatible API.
    """
    base_url_raw = kwargs.pop("base_url", None)
    if not base_url_raw:
        base_url_raw = os.getenv("LLM_BASE_URL", "http://localhost:6969")
    base_url = _normalize_openai_base_url(base_url_raw)
    
    model = kwargs.pop("model", None) or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = (
        kwargs.pop("api_key", None)
        or os.getenv("LLM_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or "sk-41bb5a29c07d4b23ad5e8e54a658ce2b"
    )
    
    return OpenAIEmbeddings(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=150,
        **kwargs
    )

__all__ = ['get_llm', 'get_embeddings']
