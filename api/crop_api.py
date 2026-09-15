"""Server-side directory crop endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_directory_service
from api.schemas import (
    DirectoryCropData,
    DirectoryCropResponse,
    PathRequest,
)
from service.directory_service import DirectoryService

router = APIRouter(prefix="/api/v1/crop", tags=["crop"])


@router.post(
    "/directory",
    response_model=DirectoryCropResponse,
    status_code=status.HTTP_200_OK,
    summary="Anonymize supported media in a server-side directory",
)
def crop_directory(
    request: PathRequest,
    service: Annotated[DirectoryService, Depends(get_directory_service)],
) -> DirectoryCropResponse:
    if not request.path:
        raise HTTPException(status_code=400, detail="Path is required")
    try:
        service.process(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Directory not found") from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail="Path must be a directory") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DirectoryCropResponse(
        status="success",
        message="Directory processed successfully",
        data=DirectoryCropData(
            directory_path=request.path,
        ),
    )
