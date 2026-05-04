"""Base loader class."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):
    """Base class for document loaders."""

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Check if this loader supports the file.

        Args:
            file_path: Path to file

        Returns:
            True if supported
        """

    @abstractmethod
    def load(self, file_path: str) -> dict:
        """
        Load document content.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with 'text', 'metadata', and optional 'images'
        """

    def _validate_file(self, file_path: str):
        """Validate that file exists and is readable."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        if not path.stat().st_size:
            raise ValueError(f"File is empty: {file_path}")
