"""Video cleaning HTTP endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import (
    get_file_manager,
    get_video_preview_service,
    get_video_service,
)
from api.schemas import ErrorResponse, VideoCleanResponse, VideoPreviewResponse
from service.file_manager import FileManager
from service.video_preview_service import VideoPreviewService
from service.video_service import VideoService
from src.errors import CleanerError, ErrorCode

router = APIRouter(prefix="/api/v1/video", tags=["video"])


@router.post(
    "/preview",
    response_model=VideoPreviewResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Convert a source video into a browser-previewable MP4",
)
def preview_video(
    file: Annotated[UploadFile, File(description="MP4 or AVI video")],
    file_manager: Annotated[FileManager, Depends(get_file_manager)],
    service: Annotated[VideoPreviewService, Depends(get_video_preview_service)],
) -> VideoPreviewResponse:
    input_path = None
    try:
        input_path = file_manager.save_upload(file.file, file.filename, "video")
        result = service.create(input_path, file_manager.preview_path(input_path))
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerError(ErrorCode.PROCESS_FAILED) from exc
    finally:
        file.file.close()
        if input_path is not None:
            file_manager.remove_upload(input_path)

    return VideoPreviewResponse(output=result.output_path.name)


@router.post(
    "/clean",
    response_model=VideoCleanResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def clean_video(
    file: Annotated[UploadFile, File(description="MP4 or AVI video")],
    file_manager: Annotated[FileManager, Depends(get_file_manager)],
    service: Annotated[VideoService, Depends(get_video_service)],
) -> VideoCleanResponse:
    try:
        input_path = file_manager.save_upload(file.file, file.filename, "video")
        output_path = file_manager.output_path(input_path, "video")
        result = service.process(input_path, output_path)
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerError(ErrorCode.PROCESS_FAILED) from exc
    finally:
        file.file.close()

    return VideoCleanResponse(
        anonymized=result.anonymized,
        frames=result.frames,
        output=result.output_path.name,
    )
