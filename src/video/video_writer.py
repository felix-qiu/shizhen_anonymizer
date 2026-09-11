"""OpenCV MP4 video output."""

from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from src.errors import CleanerError, ErrorCode


class VideoWriter:
    """Write fixed-size frames to an MP4 file using the ``mp4v`` codec."""

    CODEC = "mp4v"

    def __init__(
        self, output_path: str | Path, fps: float, size: tuple[int, int]
    ) -> None:
        self.output_path = Path(output_path)
        self.size = size
        if self.output_path.suffix.lower() != ".mp4" or fps <= 0:
            raise CleanerError(ErrorCode.VIDEO_WRITE_FAILED, str(self.output_path))
        width, height = size
        if width <= 0 or height <= 0:
            raise CleanerError(ErrorCode.VIDEO_WRITE_FAILED, "invalid output size")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.CODEC)
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, size)
        if not self._writer.isOpened():
            self._writer.release()
            raise CleanerError(ErrorCode.VIDEO_WRITE_FAILED, str(self.output_path))

    def write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self.size:
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED,
                f"frame size {(width, height)} does not match {self.size}",
            )
        try:
            self._writer.write(frame)
        except cv2.error as exc:
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED, str(self.output_path)
            ) from exc

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
