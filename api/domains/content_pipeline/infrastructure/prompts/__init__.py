"""Prompt templates."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from file.

    Args:
        prompt_name: Name of prompt file (without .txt extension)

    Returns:
        Prompt template as string
    """
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


EXTRACTION_PROMPT = load_prompt("extraction_prompt")
EVIDENCE_VERIFICATION_PROMPT = load_prompt("evidence_verification_prompt")
CYCLE_REMOVAL_PROMPT = load_prompt("cycle_removal_prompt")
