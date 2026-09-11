"""Video cleaning HTTP endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_file_manager, get_video_service
from api.schemas import ErrorResponse, VideoCleanResponse
from service.file_manager import FileManager
from service.video_service import VideoService
from src.errors import CleanerError, ErrorCode

router = APIRouter(prefix="/api/v1/video", tags=["video"])


@router.post(
    "/clean",
    response_model=VideoCleanResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def clean_video(
    file: Annotated[UploadFile, File(description="MP4 video")],
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

    return VideoCleanResponse(frames=result.frames, output=result.output_path.name)
