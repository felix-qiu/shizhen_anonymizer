"""Streaming OpenCV video frame extraction."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from src.errors import CleanerError, ErrorCode


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_index: int
    timestamp: float
    image: np.ndarray


class FrameExtractor:
    """Open a video once and yield frames without buffering the whole file."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise CleanerError(ErrorCode.VIDEO_NOT_FOUND, str(self.video_path))

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise CleanerError(ErrorCode.VIDEO_OPEN_FAILED, str(self.video_path))

        self.metadata = VideoMetadata(
            width=round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            total_frames=round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if self.metadata.width <= 0 or self.metadata.height <= 0:
            self.close()
            raise CleanerError(ErrorCode.VIDEO_OPEN_FAILED, str(self.video_path))

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def fps(self) -> float:
        return self.metadata.fps

    @property
    def total_frames(self) -> int:
        return self.metadata.total_frames

    def frames(self) -> Iterator[VideoFrame]:
        frame_index = 0
        while True:
            success, image = self._capture.read()
            if not success:
                break
            timestamp_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms > 0:
                timestamp = timestamp_ms / 1000.0
            elif self.fps > 0:
                timestamp = frame_index / self.fps
            else:
                timestamp = 0.0
            yield VideoFrame(frame_index, timestamp, image)
            frame_index += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
