"""Browser-compatible preview generation for uploaded videos."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.errors import CleanerError, ErrorCode


@dataclass(frozen=True, slots=True)
class VideoPreviewResult:
    output_path: Path


class VideoPreviewService:
    """Transcode source videos to browser-playable H.264 MP4 previews."""

    def __init__(
        self, executable: str = "ffmpeg", logger: logging.Logger | None = None
    ) -> None:
        self._executable = executable
        self._logger = logger or logging.getLogger(__name__)

    def create(
        self, input_path: str | Path, output_path: str | Path
    ) -> VideoPreviewResult:
        source = Path(input_path)
        destination = Path(output_path)
        try:
            completed = subprocess.run(
                [
                    self._executable,
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-movflags",
                    "+faststart",
                    str(destination),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CleanerError(ErrorCode.PROCESS_FAILED, "video preview failed") from exc

        if completed.returncode != 0 or not destination.is_file():
            destination.unlink(missing_ok=True)
            raise CleanerError(ErrorCode.INVALID_FILE, "video preview failed")

        self._logger.info(
            "video_preview_created input=%s output=%s",
            source.name,
            destination.name,
        )
        return VideoPreviewResult(output_path=destination)
