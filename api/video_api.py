"""Video cleaning HTTP endpoint."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import (
    get_file_manager,
    get_image_service,
    get_video_preview_service,
    get_video_service,
)
from api.schemas import (
    ErrorResponse,
    ImageCheckResponse,
    VideoCheckResponse,
    VideoCleanResponse,
    VideoPreviewResponse,
)
from service.file_manager import FileManager
from service.image_service import ImageService
from service.video_preview_service import VideoPreviewService
from service.video_service import VideoService
from src.errors import CleanerError, ErrorCode

router = APIRouter(prefix="/api/v1/video", tags=["video"])


@router.post(
    "/check",
    response_model=ImageCheckResponse | VideoCheckResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Check whether an image or video needs top-region anonymization",
)
def check_media(
    file: Annotated[
        UploadFile, File(description="JPG, JPEG, PNG, BMP, MP4, AVI, or WMV file")
    ],
    file_manager: Annotated[FileManager, Depends(get_file_manager)],
    image_service: Annotated[ImageService, Depends(get_image_service)],
    video_service: Annotated[VideoService, Depends(get_video_service)],
) -> ImageCheckResponse | VideoCheckResponse:
    input_path = None
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix in FileManager.IMAGE_EXTENSIONS:
            input_path = file_manager.save_upload(file.file, file.filename, "image")
            image_result = image_service.check(input_path)
            return ImageCheckResponse(
                needs_anonymization=image_result.needs_anonymization,
                confidence=image_result.confidence,
                bbox=image_result.bbox,
                top_crop_pixels=image_result.top_crop_pixels,
                top_crop_ratio=image_result.top_crop_ratio,
            )
        if suffix not in FileManager.VIDEO_EXTENSIONS:
            raise CleanerError(ErrorCode.INVALID_FILE)

        input_path = file_manager.save_upload(file.file, file.filename, "video")
        video_result = video_service.check(input_path)
        return VideoCheckResponse(
            needs_anonymization=video_result.needs_anonymization,
            confidence=video_result.confidence,
            bbox=video_result.bbox,
            top_crop_pixels=video_result.top_crop_pixels,
            sampled_frames=video_result.sampled_frames,
            detected_frames=video_result.detected_frames,
            failed_frames=video_result.failed_frames,
        )
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerError(ErrorCode.PROCESS_FAILED) from exc
    finally:
        file.file.close()
        if input_path is not None:
            file_manager.remove_upload(input_path)


@router.post(
    "/preview",
    response_model=VideoPreviewResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Convert a source video into a browser-previewable MP4",
)
def preview_video(
    file: Annotated[UploadFile, File(description="MP4, AVI, or WMV video")],
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
    file: Annotated[UploadFile, File(description="MP4, AVI, or WMV video")],
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
