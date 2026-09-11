"""Stable application error codes used by the command-line workflow."""

from enum import StrEnum


class ErrorCode(StrEnum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_FILE = "INVALID_FILE"
    PROCESS_FAILED = "PROCESS_FAILED"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    IMAGE_LOAD_FAILED = "IMAGE_LOAD_FAILED"
    ROI_NOT_FOUND = "ROI_NOT_FOUND"
    VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
    VIDEO_OPEN_FAILED = "VIDEO_OPEN_FAILED"
    VIDEO_WRITE_FAILED = "VIDEO_WRITE_FAILED"
    FRAME_PROCESS_FAILED = "FRAME_PROCESS_FAILED"


class CleanerError(RuntimeError):
    def __init__(self, code: ErrorCode, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        message = code.value if detail is None else f"{code.value}: {detail}"
        super().__init__(message)
