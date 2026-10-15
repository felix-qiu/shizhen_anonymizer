"""Streaming OpenCV video frame extraction."""

import json
import subprocess
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

    @staticmethod
    def _probe_wmv_timing(video_path: Path) -> tuple[float, int] | None:
        """Recover average WMV FPS when OpenCV exposes the ASF time base as FPS."""

        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration,nb_read_frames",
                    "-of",
                    "json",
                    str(video_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if completed.returncode != 0:
                return None
            streams = json.loads(completed.stdout).get("streams", [])
            if not streams:
                return None
            duration = float(streams[0].get("duration", 0))
            frame_count = int(streams[0].get("nb_read_frames", 0))
            if duration <= 0 or frame_count <= 0:
                return None
            return frame_count / duration, frame_count
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, json.JSONDecodeError):
            return None

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise CleanerError(ErrorCode.VIDEO_NOT_FOUND, str(self.video_path))

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise CleanerError(ErrorCode.VIDEO_OPEN_FAILED, str(self.video_path))

        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        total_frames = round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.video_path.suffix.lower() == ".wmv":
            probed_timing = self._probe_wmv_timing(self.video_path)
            if probed_timing is not None:
                fps, total_frames = probed_timing

        self.metadata = VideoMetadata(
            width=round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            total_frames=total_frames,
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
