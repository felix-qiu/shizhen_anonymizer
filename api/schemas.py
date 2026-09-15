"""Public Phase 4 API response schemas."""

from pydantic import BaseModel


class ImageCleanResponse(BaseModel):
    success: bool = True
    confidence: float
    bbox: list[int]
    output: str


class ImageCheckResponse(BaseModel):
    success: bool = True
    needs_anonymization: bool
    confidence: float
    bbox: list[int]
    top_crop_pixels: int
    top_crop_ratio: float


class VideoCleanResponse(BaseModel):
    success: bool = True
    frames: int
    output: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
