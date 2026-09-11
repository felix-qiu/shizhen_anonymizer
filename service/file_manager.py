"""Safe uploaded-file persistence and unique output naming."""

from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfileobj
from typing import BinaryIO, ClassVar
from uuid import uuid4

from src.errors import CleanerError, ErrorCode


class FileManager:
    IMAGE_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".bmp"}
    VIDEO_EXTENSIONS: ClassVar[set[str]] = {".mp4"}

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _unique_stem() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{uuid4().hex}"

    def save_upload(
        self, source: BinaryIO, original_name: str | None, media_kind: str
    ) -> Path:
        suffix = Path(original_name or "").suffix.lower()
        allowed = (
            self.IMAGE_EXTENSIONS if media_kind == "image" else self.VIDEO_EXTENSIONS
        )
        if suffix not in allowed:
            raise CleanerError(ErrorCode.INVALID_FILE)

        destination = self.input_dir / f"{self._unique_stem()}{suffix}"
        try:
            with destination.open("xb") as output:
                copyfileobj(source, output, length=1024 * 1024)
        except OSError as exc:
            raise CleanerError(ErrorCode.PROCESS_FAILED) from exc
        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise CleanerError(ErrorCode.INVALID_FILE)
        return destination

    def output_path(self, input_path: str | Path, media_kind: str) -> Path:
        path = Path(input_path)
        suffix = ".png" if media_kind == "image" else ".mp4"
        return self.output_dir / f"{path.stem}_clean{suffix}"
