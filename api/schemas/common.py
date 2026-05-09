from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, ConfigDict

DataT = TypeVar("DataT")

class BaseStandardModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

class StandardResponse(BaseStandardModel, Generic[DataT]):
    success: bool = True
    data: DataT
    meta: Optional[dict[str, Any]] = None

class ErrorDetail(BaseStandardModel):
    code: str
    message: str
    detail: Optional[str] = None
    meta: Optional[Any] = None

class StandardErrorResponse(BaseStandardModel):
    success: bool = False
    error: ErrorDetail
