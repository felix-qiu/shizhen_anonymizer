"""FastAPI dependency accessors backed by application state."""

from typing import cast

from fastapi import Request

from service.file_manager import FileManager
from service.image_service import ImageService
from service.video_service import VideoService
from src.errors import CleanerError, ErrorCode


def _state_value(request: Request, name: str) -> object:
    value = getattr(request.app.state, name, None)
    if value is None:
        raise CleanerError(ErrorCode.PROCESS_FAILED)
    return value


def get_file_manager(request: Request) -> FileManager:
    return cast("FileManager", _state_value(request, "file_manager"))


def get_image_service(request: Request) -> ImageService:
    return cast("ImageService", _state_value(request, "image_service"))


def get_video_service(request: Request) -> VideoService:
    return cast("VideoService", _state_value(request, "video_service"))
