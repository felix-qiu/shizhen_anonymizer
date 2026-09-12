"""Safe access to generated image and video files."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api.dependencies import get_file_manager
from service.file_manager import FileManager
from src.errors import CleanerError, ErrorCode

router = APIRouter(prefix="/api/v1/output", tags=["output"])

MEDIA_TYPES = {
    ".png": "image/png",
    ".mp4": "video/mp4",
}


@router.get("/{filename}", response_class=FileResponse)
def get_output_file(
    filename: str,
    file_manager: Annotated[FileManager, Depends(get_file_manager)],
) -> FileResponse:
    """Return one generated file without exposing arbitrary filesystem paths."""
    requested = Path(filename)
    if requested.name != filename or requested.suffix.lower() not in MEDIA_TYPES:
        raise CleanerError(ErrorCode.FILE_NOT_FOUND)

    output_root = file_manager.output_dir.resolve()
    output_path = (output_root / requested.name).resolve()
    if output_path.parent != output_root or not output_path.is_file():
        raise CleanerError(ErrorCode.FILE_NOT_FOUND)

    return FileResponse(
        output_path,
        media_type=MEDIA_TYPES[output_path.suffix.lower()],
        filename=output_path.name,
        content_disposition_type="inline",
    )
