"""Video-cleaning application service."""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.detector.roi_detector import ROIDetector
from src.processor.crop_engine import OutputSize
from src.video.video_processor import VideoProcessor


@dataclass(frozen=True, slots=True)
class VideoServiceResult:
    frames: int
    confidence: float
    inference_time_ms: float
    total_time_seconds: float
    output_path: Path


class VideoService:
    def __init__(
        self,
        detector: ROIDetector,
        detect_interval: int,
        output_size: OutputSize | None,
        output_fps: float | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._processor = VideoProcessor(
            detector=detector,
            detect_interval=detect_interval,
            output_size=output_size,
            output_fps=output_fps,
        )
        self._logger = logger or logging.getLogger(__name__)

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
            frames=result.processed_frames,
            confidence=result.initial_roi.confidence,
            inference_time_ms=result.inference_time_ms,
            total_time_seconds=result.total_time_seconds,
            output_path=destination,
        )
