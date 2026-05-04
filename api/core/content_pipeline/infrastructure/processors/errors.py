"""Custom exceptions for IO operations."""


class LoaderNotFoundError(Exception):
    """Raised when no suitable loader is found for a file."""


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""


class ChunkingError(Exception):
    """Raised when chunking fails."""
