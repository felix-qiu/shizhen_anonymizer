"""FastAPI application entry point and service lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.crop_api import router as crop_router
from api.image_api import router as image_router
from api.logging_config import configure_logging
from api.output_api import router as output_router
from api.schemas import ErrorResponse
from api.video_api import router as video_router
from service.dicom_service import DicomService
from service.directory_service import DirectoryService
from service.file_manager import FileManager
from service.image_service import ImageService
from service.server_config import ServerSettings, load_server_settings
from service.video_service import VideoService
from src.config import load_config
from src.errors import CleanerError, ErrorCode
from src.model.model_manager import ModelManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
ERROR_STATUS = {
    ErrorCode.FILE_NOT_FOUND: 404,
    ErrorCode.MODEL_NOT_FOUND: 503,
    ErrorCode.INVALID_FILE: 400,
    ErrorCode.ROI_NOT_FOUND: 422,
    ErrorCode.PROCESS_FAILED: 500,
}


def _api_error_code(code: ErrorCode) -> ErrorCode:
    if code in {ErrorCode.IMAGE_NOT_FOUND, ErrorCode.VIDEO_NOT_FOUND}:
        return ErrorCode.FILE_NOT_FOUND
    if code in {ErrorCode.IMAGE_LOAD_FAILED, ErrorCode.VIDEO_OPEN_FAILED}:
        return ErrorCode.INVALID_FILE
    if code in {ErrorCode.VIDEO_WRITE_FAILED, ErrorCode.FRAME_PROCESS_FAILED}:
        return ErrorCode.PROCESS_FAILED
    return code


def create_app(
    settings: ServerSettings | None = None,
    *,
    load_model_on_startup: bool = True,
) -> FastAPI:
    server_settings = settings or load_server_settings(
        PROJECT_ROOT / "config/server.yaml", PROJECT_ROOT
    )
    runtime_config = load_config(PROJECT_ROOT / "config/config.yaml", PROJECT_ROOT)
    logger = configure_logging(server_settings.log_file)
    file_manager = FileManager(server_settings.input_dir, server_settings.output_dir)
    model_manager = ModelManager(
        model_path=str(server_settings.model_path),
        confidence_threshold=runtime_config.confidence_threshold,
        device=runtime_config.device,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        dicom_service: DicomService | None = None
        if load_model_on_startup:
            detector = model_manager.load()
            application.state.image_service = ImageService(
                detector, runtime_config.output_size, logger
            )
            application.state.video_service = VideoService(
                detector,
                runtime_config.video.detect_interval,
                runtime_config.output_size,
                runtime_config.video.output_fps,
                logger,
            )
            dicom_service = DicomService(
                server_settings.dicom_server_url,
                server_settings.dicom_timeout_seconds,
                logger,
            )
            application.state.directory_service = DirectoryService(
                application.state.image_service,
                application.state.video_service,
                dicom_service,
                logger,
            )
            logger.info("model_loaded path=%s", server_settings.model_path)
        try:
            yield
        finally:
            if dicom_service is not None:
                dicom_service.close()
            model_manager.close()
            logger.info("service_stopped")

    application = FastAPI(
        title="shizhen_anonymizer",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.file_manager = file_manager
    application.state.model_manager = model_manager
    application.state.logger = logger
    application.include_router(image_router)
    application.include_router(video_router)
    application.include_router(crop_router)
    application.include_router(output_router)
    application.mount(
        "/assets",
        StaticFiles(directory=WEB_ROOT / "assets"),
        name="web-assets",
    )

    @application.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html", media_type="text/html")

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.middleware("http")
    async def request_logging(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = perf_counter()
        response = await call_next(request)
        logger.info(
            "request method=%s path=%s status=%d duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response

    @application.exception_handler(CleanerError)
    async def cleaner_error_handler(
        request: Request, exc: CleanerError
    ) -> JSONResponse:
        public_code = _api_error_code(exc.code)
        logger.warning(
            "request_failed path=%s error=%s", request.url.path, public_code.value
        )
        payload = ErrorResponse(error=public_code.value)
        return JSONResponse(
            status_code=ERROR_STATUS.get(public_code, 500),
            content=payload.model_dump(),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "request_invalid path=%s details=%s", request.url.path, exc.errors()
        )
        payload = ErrorResponse(error=ErrorCode.INVALID_FILE.value)
        return JSONResponse(status_code=400, content=payload.model_dump())

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("request_failed path=%s error=%r", request.url.path, exc)
        payload = ErrorResponse(error=ErrorCode.PROCESS_FAILED.value)
        return JSONResponse(status_code=500, content=payload.model_dump())

    return application


app = create_app()
