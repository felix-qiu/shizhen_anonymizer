"""Browser-compatible MP4 video output."""

import os
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self
from uuid import uuid4

import numpy as np

from src.errors import CleanerError, ErrorCode


class VideoWriter:
    """Stream raw OpenCV frames directly into one H.264 encoding pass."""

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
        token = uuid4().hex
        self._encoded_path = self.output_path.with_name(
            f".{self.output_path.stem}.{token}.h264.mp4"
        )
        self._closed = False
        self._stderr: BinaryIO = tempfile.TemporaryFile()  # noqa: SIM115
        try:
            self._process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self._encoded_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except OSError as exc:
            self._stderr.close()
            self._encoded_path.unlink(missing_ok=True)
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED, str(self.output_path)
            ) from exc

    def write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self.size:
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED,
                f"frame size {(width, height)} does not match {self.size}",
            )
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED, "frame must be uint8 BGR"
            )
        try:
            if self._process.stdin is None:
                raise BrokenPipeError
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED, self._error_detail()
            ) from exc

    def _error_detail(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        lines = self._stderr.read().decode("utf-8", errors="replace").splitlines()
        return lines[-1] if lines else str(self.output_path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass
            return_code = self._process.wait(timeout=300)
            if return_code != 0 or not self._encoded_path.is_file():
                raise CleanerError(
                    ErrorCode.VIDEO_WRITE_FAILED, self._error_detail()
                )
            os.replace(self._encoded_path, self.output_path)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            self._process.wait()
            raise CleanerError(
                ErrorCode.VIDEO_WRITE_FAILED, str(self.output_path)
            ) from exc
        finally:
            self._stderr.close()
            self._encoded_path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
