"""Shared API contract schemas.

Domain-specific request/response models now live in each domain package
(``api.domains.<domain>.schemas``); this package keeps only the cross-cutting
envelope, enums, validators, and the content-pipeline schemas.
"""

from .common import (
    ErrorDetail,
    InfoResponse,
    ReadinessResponse,
    StandardErrorResponse,
    StandardResponse,
)

__all__ = [
    "ErrorDetail",
    "InfoResponse",
    "ReadinessResponse",
    "StandardErrorResponse",
    "StandardResponse",
]
