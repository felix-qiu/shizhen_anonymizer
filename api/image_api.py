"""Image cleaning HTTP endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_file_manager, get_image_service
from api.schemas import ErrorResponse, ImageCheckResponse, ImageCleanResponse
from service.file_manager import FileManager
from service.image_service import ImageService
from src.errors import CleanerError, ErrorCode

router = APIRouter(prefix="/api/v1/image", tags=["image"])


@router.post(
    "/check",
    response_model=ImageCheckResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Check whether an image needs top-region anonymization",
)
def check_image(
    file: Annotated[UploadFile, File(description="JPG, PNG, or BMP image")],
    file_manager: Annotated[FileManager, Depends(get_file_manager)],
    service: Annotated[ImageService, Depends(get_image_service)],
) -> ImageCheckResponse:
    input_path = None
    try:
        input_path = file_manager.save_upload(file.file, file.filename, "image")
        result = service.check(input_path)
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerError(ErrorCode.PROCESS_FAILED) from exc
    finally:
        file.file.close()
        if input_path is not None:
            file_manager.remove_upload(input_path)

    return ImageCheckResponse(
        needs_anonymization=result.needs_anonymization,
        confidence=result.confidence,
        bbox=result.bbox,
        top_crop_pixels=result.top_crop_pixels,
        top_crop_ratio=result.top_crop_ratio,
    )


@router.post(
    "/clean",
    response_model=ImageCleanResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def clean_image(
    file: Annotated[UploadFile, File(description="JPG, PNG, or BMP image")],
    file_manager: Annotated[FileManager, Depends(get_file_manager)],
    service: Annotated[ImageService, Depends(get_image_service)],
) -> ImageCleanResponse:
    try:
        input_path = file_manager.save_upload(file.file, file.filename, "image")
        output_path = file_manager.output_path(input_path, "image")
        result = service.process(input_path, output_path)
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerError(ErrorCode.PROCESS_FAILED) from exc
    finally:
        file.file.close()

    return ImageCleanResponse(
        confidence=result.confidence,
        bbox=result.bbox,
        output=result.output_path.name,
    )
