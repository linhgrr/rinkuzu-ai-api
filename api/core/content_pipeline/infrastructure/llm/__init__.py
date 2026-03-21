from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .....config import get_settings


def _normalize_openai_base_url(url: str) -> str:
    """Ensure OpenAI-compatible base URL includes /v1 prefix path."""
    raw = (url or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def _resolve_llm_api_key() -> str:
    settings = get_settings()
    return (
        settings.llm_api_key
        or settings.gemini_api_key
        or settings.google_api_key
        or "sk-41bb5a29c07d4b23ad5e8e54a658ce2b"
    )


def get_llm(temperature=0.0, **kwargs):
    """
    Returns a configured ChatOpenAI instance pointing to a local compatible API.
    """
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None)
    if not base_url_raw:
        base_url_raw = settings.llm_base_url or "http://localhost:6969"
    base_url = _normalize_openai_base_url(base_url_raw)

    model = kwargs.pop("model", None) or settings.llm_model or "gemini-3.0-pro"
    api_key = kwargs.pop("api_key", None) or _resolve_llm_api_key()

    # Remove kwargs that are specific to ChatGoogleGenerativeAI
    kwargs.pop("max_retries", None)

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_sec,
        **kwargs
    )


def get_embeddings(**kwargs):
    """
    Returns a configured OpenAIEmbeddings instance pointing to a local compatible API.
    """
    settings = get_settings()
    base_url_raw = kwargs.pop("base_url", None)
    if not base_url_raw:
        base_url_raw = settings.llm_base_url or "http://localhost:6969"
    base_url = _normalize_openai_base_url(base_url_raw)

    model = kwargs.pop("model", None) or settings.llm_embedding_model
    api_key = kwargs.pop("api_key", None) or _resolve_llm_api_key()

    return OpenAIEmbeddings(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=settings.llm_timeout_sec,
        **kwargs
    )

__all__ = ['get_llm', 'get_embeddings']
