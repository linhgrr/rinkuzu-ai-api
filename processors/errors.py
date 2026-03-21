"""Custom exceptions for IO operations."""


class LoaderNotFoundError(Exception):
    """Raised when no suitable loader is found for a file."""
    pass


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""
    pass


class ChunkingError(Exception):
    """Raised when chunking fails."""
    pass
