---
name: llm
description: "Skill for the LLM area of rinkuzu-ai-api after the shared LiteLLM migration."
---

# LLM

This area is now standardized around the shared LiteLLM-backed abstraction in `api/core/shared/llm.py`.

## When to Use

- Working with code in `api/`
- Modifying provider configuration, structured generation, or streaming text generation
- Understanding how the backend normalizes provider-specific message shapes into the project-wide LLM surface

## Key Files

| File | Purpose |
|------|---------|
| `api/core/shared/llm.py` | Shared provider config, chat-completions adapter, text + structured generation helpers |
| `api/core/shared/document_text.py` | Provider boundary for page-level document text extraction |
| `api/core/content_pipeline/infrastructure/llm/structured_generation.py` | Content-pipeline structured generation adapter |
| `api/core/content_pipeline/infrastructure/llm/schemas.py` | Pydantic schemas used for structured extraction |
| `api/core/content_pipeline/infrastructure/llm/postprocess.py` | Post-processing for extracted concepts |

## Entry Points

- `build_llm_provider_config`
- `LiteLLMClient`
- `invoke_text_completion`
- `astream_text_completion`
- `invoke_structured_completion`
- `ainvoke_structured_completion`
- `LiteLLMStructuredGenerationClient.parse_response`

## Notes

- Legacy file-upload / Responses-specific extraction paths have been removed.
- Runtime config is strict: use `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`; the model string is passed through as-is.
- Runtime LLM consumers should go through `api.core.shared.llm` instead of creating provider clients directly.
