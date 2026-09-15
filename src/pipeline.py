"""Detector-independent image cleaning business workflow."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.detector.roi_detector import ROIDetector, ROIResult
from src.errors import CleanerError, ErrorCode
from src.processor.crop_engine import OutputSize, crop_image, top_boundary_bbox


@dataclass(frozen=True, slots=True)
class CleanResult:
    image: np.ndarray
    roi: ROIResult
    inference_time_ms: float


@dataclass(frozen=True, slots=True)
class ROIInspectionResult:
    """Detected effective ROI without modifying the source image."""

    roi: ROIResult
    inference_time_ms: float


class ImageCleaner:
    """Run ROI detection and standardized cropping using any detector backend."""

    def __init__(self, detector: ROIDetector, output_size: OutputSize | None) -> None:
        self.detector = detector
        self.output_size = output_size

    def inspect(self, image: np.ndarray) -> ROIInspectionResult:
        """Detect and normalize the top-boundary ROI without cropping."""

        started = perf_counter()
        roi = self.detector.detect(image)
        elapsed_ms = (perf_counter() - started) * 1000
        if roi is None:
            raise CleanerError(ErrorCode.ROI_NOT_FOUND)
        effective_roi = ROIResult(
            confidence=roi.confidence,
            bbox=[round(value) for value in top_boundary_bbox(image, roi.bbox)],
        )
        return ROIInspectionResult(effective_roi, elapsed_ms)

    def clean(self, image: np.ndarray) -> CleanResult:
        inspection = self.inspect(image)
        cleaned = crop_image(image, inspection.roi.bbox, self.output_size)
        return CleanResult(cleaned, inspection.roi, inspection.inference_time_ms)
