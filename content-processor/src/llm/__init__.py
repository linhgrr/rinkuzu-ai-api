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
    base_url = _normalize_openai_base_url(
        os.getenv("LLM_BASE_URL", "http://localhost:6969")
    )
    model = os.getenv("LLM_MODEL", "gemini-3-pro-high")
    api_key = os.getenv("LLM_API_KEY", "sk-41bb5a29c07d4b23ad5e8e54a658ce2b")
    
    # Remove kwargs that are specific to ChatGoogleGenerativeAI
    kwargs.pop("max_retries", None)

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
        **kwargs
    )

def get_embeddings(**kwargs):
    """
    Returns a configured OpenAIEmbeddings instance pointing to a local compatible API.
    """
    base_url = _normalize_openai_base_url(
        os.getenv("LLM_BASE_URL", "http://localhost:6969")
    )
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = os.getenv("LLM_API_KEY", "sk-41bb5a29c07d4b23ad5e8e54a658ce2b")
    
    return OpenAIEmbeddings(
        base_url=base_url,
        model=model,
        api_key=api_key,
        **kwargs
    )

__all__ = ['get_llm', 'get_embeddings']
