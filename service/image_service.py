"""Image-cleaning application service."""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.detector.roi_detector import ROIDetector
from src.image_io import load_image, save_image
from src.pipeline import ImageCleaner
from src.processor.crop_engine import OutputSize


@dataclass(frozen=True, slots=True)
class ImageServiceResult:
    confidence: float
    bbox: list[int]
    inference_time_ms: float
    output_path: Path


@dataclass(frozen=True, slots=True)
class ImageCheckResult:
    needs_anonymization: bool
    confidence: float
    bbox: list[int]
    top_crop_pixels: int
    top_crop_ratio: float
    inference_time_ms: float


class ImageService:
    def __init__(
        self,
        detector: ROIDetector,
        output_size: OutputSize | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cleaner = ImageCleaner(detector, output_size)
        self._logger = logger or logging.getLogger(__name__)

    def check(self, input_path: str | Path) -> ImageCheckResult:
        """Determine whether the model recommends removing a top region."""

        image = load_image(input_path)
        inspection = self._cleaner.inspect(image)
        top_crop_pixels = inspection.roi.bbox[1]
        top_crop_ratio = top_crop_pixels / image.shape[0]
        needs_anonymization = top_crop_pixels > 0
        self._logger.info(
            "image_checked input=%s inference_ms=%.2f confidence=%.4f "
            "needs_anonymization=%s top_crop_pixels=%d",
            Path(input_path).name,
            inspection.inference_time_ms,
            inspection.roi.confidence,
            needs_anonymization,
            top_crop_pixels,
        )
        return ImageCheckResult(
            needs_anonymization=needs_anonymization,
            confidence=inspection.roi.confidence,
            bbox=inspection.roi.bbox,
            top_crop_pixels=top_crop_pixels,
            top_crop_ratio=top_crop_ratio,
            inference_time_ms=inspection.inference_time_ms,
        )

    def process(
        self, input_path: str | Path, output_path: str | Path
    ) -> ImageServiceResult:
        image = load_image(input_path)
        result = self._cleaner.clean(image)
        destination = Path(output_path)
        save_image(destination, result.image)
        self._logger.info(
            "image_processed input=%s inference_ms=%.2f confidence=%.4f output=%s",
            Path(input_path).name,
            result.inference_time_ms,
            result.roi.confidence,
            destination.name,
        )
        return ImageServiceResult(
            confidence=result.roi.confidence,
            bbox=result.roi.bbox,
            inference_time_ms=result.inference_time_ms,
            output_path=destination,
        )
