"""Streaming video ROI detection, reuse, cropping, and output workflow."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.detector.roi_detector import ROIDetector, ROIResult
from src.errors import CleanerError, ErrorCode
from src.processor.crop_engine import OutputSize, crop_image, top_boundary_bbox
from src.video.frame_extractor import FrameExtractor
from src.video.video_writer import VideoWriter


@dataclass(frozen=True, slots=True)
class VideoProcessResult:
    input_fps: float
    output_fps: float
    source_frames: int
    processed_frames: int
    detection_count: int
    inference_time_ms: float
    initial_roi: ROIResult
    total_time_seconds: float


class VideoProcessor:
    """Process video frames while periodically refreshing one reusable ROI."""

    def __init__(
        self,
        detector: ROIDetector,
        detect_interval: int = 30,
        output_size: OutputSize | None = None,
        output_fps: float | None = None,
    ) -> None:
        if detect_interval <= 0:
            raise ValueError("detect_interval must be a positive integer")
        if output_fps is not None and output_fps <= 0:
            raise ValueError("output_fps must be positive or None")
        self.detector = detector
        self.detect_interval = detect_interval
        self.output_size = output_size
        self.output_fps = output_fps

    def process(
        self, input_video: str | Path, output_video: str | Path
    ) -> VideoProcessResult:
        started = perf_counter()
        processed_frames = 0
        detection_count = 0
        inference_time_ms = 0.0
        current_roi: ROIResult | None = None
        initial_roi: ROIResult | None = None
        writer: VideoWriter | None = None

        with FrameExtractor(input_video) as extractor:
            effective_fps = self.output_fps or extractor.fps
            if effective_fps <= 0:
                raise CleanerError(
                    ErrorCode.VIDEO_OPEN_FAILED, "video FPS is unavailable"
                )
            try:
                for video_frame in extractor.frames():
                    try:
                        if video_frame.frame_index % self.detect_interval == 0:
                            detection_count += 1
                            inference_started = perf_counter()
                            detected_roi = self.detector.detect(video_frame.image)
                            inference_time_ms += (
                                perf_counter() - inference_started
                            ) * 1000
                            if detected_roi is not None:
                                current_roi = ROIResult(
                                    confidence=detected_roi.confidence,
                                    bbox=[
                                        round(value)
                                        for value in top_boundary_bbox(
                                            video_frame.image, detected_roi.bbox
                                        )
                                    ],
                                )
                                if initial_roi is None:
                                    initial_roi = current_roi
                        if current_roi is None:
                            raise CleanerError(ErrorCode.ROI_NOT_FOUND)

                        processed = crop_image(
                            video_frame.image, current_roi.bbox, self.output_size
                        )
                        if writer is None:
                            height, width = processed.shape[:2]
                            writer = VideoWriter(
                                output_video, effective_fps, (width, height)
                            )
                        elif (processed.shape[1], processed.shape[0]) != writer.size:
                            processed = crop_image(
                                video_frame.image, current_roi.bbox, writer.size
                            )
                        writer.write(processed)
                        processed_frames += 1
                    except CleanerError:
                        raise
                    except Exception as exc:
                        raise CleanerError(
                            ErrorCode.FRAME_PROCESS_FAILED,
                            f"frame {video_frame.frame_index}",
                        ) from exc
            finally:
                if writer is not None:
                    writer.close()

            if processed_frames == 0 or initial_roi is None:
                raise CleanerError(ErrorCode.ROI_NOT_FOUND)

            return VideoProcessResult(
                input_fps=extractor.fps,
                output_fps=effective_fps,
                source_frames=extractor.total_frames,
                processed_frames=processed_frames,
                detection_count=detection_count,
                inference_time_ms=inference_time_ms,
                initial_roi=initial_roi,
                total_time_seconds=perf_counter() - started,
            )
