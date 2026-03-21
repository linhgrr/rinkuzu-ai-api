"""MIME type detection utilities."""

import mimetypes
from pathlib import Path
from typing import Optional

mimetypes.init()

def guess_mime_type(file_path: str) -> Optional[str]:
    """
    Guess MIME type from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string or None if unknown
    """
    path = Path(file_path)
    
    mime_type, _ = mimetypes.guess_type(str(path))
    
    if mime_type:
        return mime_type
    
    suffix = path.suffix.lower()
    custom_types = {
        '.pdf': 'application/pdf',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.json': 'application/json',
        '.yaml': 'application/x-yaml',
        '.yml': 'application/x-yaml',
    }
    
    return custom_types.get(suffix)


def get_file_type(file_path: str) -> str:
    """
    Get simplified file type.
    
    Args:
        file_path: Path to file
        
    Returns:
        File type: 'pdf', 'pptx', 'video', 'text', or 'unknown'
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    type_mapping = {
        '.pdf': 'pdf',
        '.pptx': 'pptx',
        '.ppt': 'pptx',
        '.mp4': 'video',
        '.avi': 'video',
        '.mov': 'video',
        '.mkv': 'video',
        '.txt': 'text',
        '.md': 'text',
        '.doc': 'doc',
        '.docx': 'doc',
    }
    
    return type_mapping.get(suffix, 'unknown')
