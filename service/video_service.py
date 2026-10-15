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
    sampled_frames: int
    detected_frames: int
    failed_frames: int


class VideoService:
    def __init__(
        self,
        detector: ROIDetector,
        detect_interval: int,
        output_size: OutputSize | None,
        output_fps: float | None,
        logger: logging.Logger | None = None,
        top_boundary_tolerance_pixels: int = 5,
        check_interval: int = 5,
    ) -> None:
        if check_interval <= 0:
            raise ValueError("check_interval must be a positive integer")
        self._detector = detector
        self._top_boundary_tolerance_pixels = top_boundary_tolerance_pixels
        self._check_interval = check_interval
        self._processor = VideoProcessor(
            detector=detector,
            detect_interval=detect_interval,
            output_size=output_size,
            output_fps=output_fps,
            top_boundary_tolerance_pixels=top_boundary_tolerance_pixels,
        )
        self._logger = logger or logging.getLogger(__name__)

    def check(self, input_path: str | Path) -> VideoCheckResult:
        """Inspect interval frames plus the last frame without writing output."""

        sampled_frames = 0
        detected_frames = 0
        failed_frames = 0
        inference_time_ms = 0.0
        best: tuple[int, float, list[int]] | None = None
        last_frame = None
        last_sampled_index = -1
        with FrameExtractor(input_path) as extractor:
            for frame in extractor.frames():
                last_frame = frame
                if frame.frame_index % self._check_interval != 0:
                    continue
                (
                    sampled_frames,
                    detected_frames,
                    failed_frames,
                    inference_time_ms,
                    best,
                ) = self._inspect_check_frame(
                    frame.image,
                    sampled_frames,
                    detected_frames,
                    failed_frames,
                    inference_time_ms,
                    best,
                )
                last_sampled_index = frame.frame_index

            if last_frame is None:
                raise CleanerError(ErrorCode.VIDEO_OPEN_FAILED, str(input_path))

            if last_frame.frame_index != last_sampled_index:
                (
                    sampled_frames,
                    detected_frames,
                    failed_frames,
                    inference_time_ms,
                    best,
                ) = self._inspect_check_frame(
                    last_frame.image,
                    sampled_frames,
                    detected_frames,
                    failed_frames,
                    inference_time_ms,
                    best,
                )

        if best is None or (best[0] == 0 and failed_frames > 0):
            raise CleanerError(ErrorCode.ROI_NOT_FOUND)

        top_crop_pixels, confidence, bbox = best
        needs_anonymization = top_crop_pixels > 0
        self._logger.info(
            "video_checked input=%s inference_ms=%.2f confidence=%.4f "
            "needs_anonymization=%s top_crop_pixels=%d sampled=%d detected=%d "
            "failed=%d",
            Path(input_path).name,
            inference_time_ms,
            confidence,
            needs_anonymization,
            top_crop_pixels,
            sampled_frames,
            detected_frames,
            failed_frames,
        )
        return VideoCheckResult(
            needs_anonymization=needs_anonymization,
            confidence=confidence,
            bbox=bbox,
            top_crop_pixels=top_crop_pixels,
            inference_time_ms=inference_time_ms,
            sampled_frames=sampled_frames,
            detected_frames=detected_frames,
            failed_frames=failed_frames,
        )

    def _inspect_check_frame(
        self,
        image,
        sampled_frames: int,
        detected_frames: int,
        failed_frames: int,
        inference_time_ms: float,
        best: tuple[int, float, list[int]] | None,
    ) -> tuple[int, int, int, float, tuple[int, float, list[int]] | None]:
        sampled_frames += 1
        started = perf_counter()
        detected = self._detector.detect(image)
        inference_time_ms += (perf_counter() - started) * 1000
        if detected is None:
            return (
                sampled_frames,
                detected_frames,
                failed_frames + 1,
                inference_time_ms,
                best,
            )

        detected_frames += 1
        bbox = [
            round(value)
            for value in top_boundary_bbox(
                image,
                detected.bbox,
                self._top_boundary_tolerance_pixels,
            )
        ]
        candidate = (bbox[1], detected.confidence, bbox)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        return (
            sampled_frames,
            detected_frames,
            failed_frames,
            inference_time_ms,
            best,
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
            anonymized=result.anonymized,
            frames=result.processed_frames,
            confidence=result.initial_roi.confidence,
            inference_time_ms=result.inference_time_ms,
            total_time_seconds=result.total_time_seconds,
            output_path=destination,
        )
