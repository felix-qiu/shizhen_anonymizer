"""Recursive server-side directory anonymization service."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from service.dicom_service import DicomService
from service.file_manager import FileManager
from service.image_service import ImageService
from service.video_service import VideoService
from src.errors import CleanerError


@dataclass(frozen=True, slots=True)
class DirectoryFileResult:
    input_path: Path
    output_path: Path | None
    media_type: str
    status: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DirectoryProcessResult:
    directory_path: Path
    output_directory: Path
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    results: list[DirectoryFileResult]


class DirectoryService:
    """Process supported media recursively while preserving relative paths."""

    VIDEO_EXTENSIONS: ClassVar[set[str]] = {".avi", ".mp4"}

    def __init__(
        self,
        image_service: ImageService,
        video_service: VideoService,
        dicom_service: DicomService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._image_service = image_service
        self._video_service = video_service
        self._dicom_service = dicom_service
        self._logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _remove_stale_outputs(output_path: Path, media_type: str) -> None:
        """Remove results left by an earlier run when cropping is not needed."""

        output_path.unlink(missing_ok=True)
        if media_type == "video":
            output_path.with_suffix(".dcm").unlink(missing_ok=True)

    def process(self, directory_path: str | Path) -> DirectoryProcessResult:
        source = Path(directory_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if not source.is_dir():
            raise NotADirectoryError(source)

        # The storage service passes a per-request working directory and expects
        # anonymized files below its ``cropped`` child directory.
        output_root = source / "cropped"
        # Keep the directory contract stable even when no file needs cropping.
        # Downstream callers can treat an empty directory as "no anonymization
        # required" instead of failing because the output path does not exist.
        output_root.mkdir(parents=True, exist_ok=True)
        files = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and output_root not in path.parents
        )
        results: list[DirectoryFileResult] = []
        processed_files = 0
        failed_files = 0
        skipped_files = 0

        for input_path in files:
            suffix = input_path.suffix.lower()
            relative_path = input_path.relative_to(source)
            output_path = output_root / relative_path
            if suffix in FileManager.IMAGE_EXTENSIONS:
                media_type = "image"
                processor = self._image_service
            elif suffix in self.VIDEO_EXTENSIONS:
                media_type = "video"
                processor = self._video_service
                if suffix != ".mp4":
                    output_path = output_path.with_suffix(".mp4")
            else:
                skipped_files += 1
                results.append(
                    DirectoryFileResult(
                        input_path=input_path,
                        output_path=None,
                        media_type="unsupported",
                        status="skipped",
                    )
                )
                continue

            try:
                inspection = processor.check(input_path)
                if not inspection.needs_anonymization:
                    self._remove_stale_outputs(output_path, media_type)
                    skipped_files += 1
                    self._logger.info(
                        "directory_file_not_required input=%s top_crop_pixels=%d",
                        input_path,
                        inspection.top_crop_pixels,
                    )
                    results.append(
                        DirectoryFileResult(
                            input_path=input_path,
                            output_path=None,
                            media_type=media_type,
                            status="not_required",
                        )
                    )
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                process_result = processor.process(input_path, output_path)
                if not process_result.anonymized:
                    self._remove_stale_outputs(output_path, media_type)
                    skipped_files += 1
                    self._logger.info(
                        "directory_file_not_required_after_processing input=%s",
                        input_path,
                    )
                    results.append(
                        DirectoryFileResult(
                            input_path=input_path,
                            output_path=None,
                            media_type=media_type,
                            status="not_required",
                        )
                    )
                    continue
                if media_type == "video" and self._dicom_service is not None:
                    dicom_output = output_path.with_suffix(".dcm")
                    converted = self._dicom_service.convert(
                        output_path,
                        dicom_output,
                        patient_id=input_path.stem,
                    )
                    if not converted:
                        self._logger.warning(
                            "dicom_conversion_unavailable video=%s", output_path
                        )
                processed_files += 1
                results.append(
                    DirectoryFileResult(
                        input_path=input_path,
                        output_path=output_path,
                        media_type=media_type,
                        status="success",
                    )
                )
            # A malformed file or codec failure must not abort the remaining batch.
            except Exception as exc:  # noqa: BLE001
                failed_files += 1
                error = exc.code.value if isinstance(exc, CleanerError) else str(exc)
                self._logger.warning(
                    "directory_file_failed input=%s error=%s", input_path, error
                )
                results.append(
                    DirectoryFileResult(
                        input_path=input_path,
                        output_path=None,
                        media_type=media_type,
                        status="failed",
                        error=error,
                    )
                )

        self._logger.info(
            "directory_processed input=%s output=%s total=%d processed=%d "
            "failed=%d skipped=%d",
            source,
            output_root,
            len(files),
            processed_files,
            failed_files,
            skipped_files,
        )
        return DirectoryProcessResult(
            directory_path=source,
            output_directory=output_root,
            total_files=len(files),
            processed_files=processed_files,
            failed_files=failed_files,
            skipped_files=skipped_files,
            results=results,
        )
