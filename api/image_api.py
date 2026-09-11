"""Image cleaning HTTP endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_file_manager, get_image_service
from api.schemas import ErrorResponse, ImageCleanResponse
from service.file_manager import FileManager
from service.image_service import ImageService
from src.errors import CleanerError, ErrorCode

router = APIRouter(prefix="/api/v1/image", tags=["image"])


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
