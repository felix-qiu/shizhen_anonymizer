"""Public API request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class PathRequest(BaseModel):
    path: str = Field(description="Absolute server-side directory path")
    file_type: Literal["image", "video", "directory"] | None = Field(
        default=None,
        description="Optional compatibility field; ignored for directory processing",
    )


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


class DirectoryCropData(BaseModel):
    directory_path: str


class DirectoryCropResponse(BaseModel):
    status: Literal["success"]
    message: str
    data: DirectoryCropData


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
