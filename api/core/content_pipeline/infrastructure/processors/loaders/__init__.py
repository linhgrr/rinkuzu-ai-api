"""Document loaders."""

from .base import BaseLoader
from .local_pdf_text_loader import LocalPdfTextLoader

__all__ = ["BaseLoader", "LocalPdfTextLoader"]
