"""Public Phase 4 API response schemas."""

from pydantic import BaseModel


class ImageCleanResponse(BaseModel):
    success: bool = True
    confidence: float
    bbox: list[int]
    output: str


class VideoCleanResponse(BaseModel):
    success: bool = True
    frames: int
    output: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
