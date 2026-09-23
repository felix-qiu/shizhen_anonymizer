"""Video-cleaning application service."""

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.detector.roi_detector import ROIDetector
from src.errors import CleanerError, ErrorCode
from src.processor.crop_engine import OutputSize, top_boundary_bbox
from src.video.frame_extractor import FrameExtractor
from src.video.video_processor import VideoProcessor


@dataclass(frozen=True, slots=True)
class VideoServiceResult:
    frames: int
    confidence: float
    inference_time_ms: float
    total_time_seconds: float
    output_path: Path
    anonymized: bool = False


@dataclass(frozen=True, slots=True)
class VideoCheckResult:
    needs_anonymization: bool
    confidence: float
    bbox: list[int]
    top_crop_pixels: int
    inference_time_ms: float


class VideoService:
    def __init__(
        self,
        detector: ROIDetector,
        detect_interval: int,
        output_size: OutputSize | None,
        output_fps: float | None,
        logger: logging.Logger | None = None,
        top_boundary_tolerance_pixels: int = 5,
    ) -> None:
        self._detector = detector
        self._top_boundary_tolerance_pixels = top_boundary_tolerance_pixels
        self._processor = VideoProcessor(
            detector=detector,
            detect_interval=detect_interval,
            output_size=output_size,
            output_fps=output_fps,
            top_boundary_tolerance_pixels=top_boundary_tolerance_pixels,
        )
        self._logger = logger or logging.getLogger(__name__)

    def check(self, input_path: str | Path) -> VideoCheckResult:
        """Inspect the first frame without creating an output video."""

        with FrameExtractor(input_path) as extractor:
            frame = next(extractor.frames(), None)
            if frame is None:
                raise CleanerError(ErrorCode.VIDEO_OPEN_FAILED, str(input_path))
            started = perf_counter()
            detected = self._detector.detect(frame.image)
            inference_time_ms = (perf_counter() - started) * 1000
            if detected is None:
                raise CleanerError(ErrorCode.ROI_NOT_FOUND)
            bbox = [
                round(value)
                for value in top_boundary_bbox(
                    frame.image,
                    detected.bbox,
                    self._top_boundary_tolerance_pixels,
                )
            ]

        top_crop_pixels = bbox[1]
        needs_anonymization = top_crop_pixels > 0
        self._logger.info(
            "video_checked input=%s inference_ms=%.2f confidence=%.4f "
            "needs_anonymization=%s top_crop_pixels=%d",
            Path(input_path).name,
            inference_time_ms,
            detected.confidence,
            needs_anonymization,
            top_crop_pixels,
        )
        return VideoCheckResult(
            needs_anonymization=needs_anonymization,
            confidence=detected.confidence,
            bbox=bbox,
            top_crop_pixels=top_crop_pixels,
            inference_time_ms=inference_time_ms,
        )

    def process(
        self, input_path: str | Path, output_path: str | Path
    ) -> VideoServiceResult:
        result = self._processor.process(input_path, output_path)
        destination = Path(output_path)
        self._logger.info(
            "video_processed input=%s inference_ms=%.2f total_seconds=%.2f "
            "confidence=%.4f frames=%d output=%s",
            Path(input_path).name,
            result.inference_time_ms,
            result.total_time_seconds,
            result.initial_roi.confidence,
            result.processed_frames,
            destination.name,
        )
        return VideoServiceResult(
            anonymized=result.initial_roi.bbox[1] > 0,
            frames=result.processed_frames,
            confidence=result.initial_roi.confidence,
            inference_time_ms=result.inference_time_ms,
            total_time_seconds=result.total_time_seconds,
            output_path=destination,
        )
