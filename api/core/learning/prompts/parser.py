"""
parser.py - Handles validation and parsing of LLM structured outputs.
"""

from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class OutputParsingError(Exception):
    """Exception raised when parsing structured output fails."""


class OutputParser:
    """Utility to parse and validate output from LLM."""

    @staticmethod
    def parse_and_validate(output: Any, schema: type[T]) -> T:
        """
        Ensures the output matches the expected Pydantic schema.
        Raises OutputParsingError if validation fails.
        """
        if isinstance(output, schema):
            return output

        if isinstance(output, dict):
            try:
                return schema(**output)
            except ValidationError as e:
                logger.error(f"Validation error for {schema.__name__}: {e}")
                raise OutputParsingError(f"Failed to parse dict to {schema.__name__}: {e}") from e

        raise OutputParsingError(
            f"Output must be of type {schema.__name__} or dict, got {type(output)}"
        )
