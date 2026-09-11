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


class ImageService:
    def __init__(
        self,
        detector: ROIDetector,
        output_size: OutputSize | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cleaner = ImageCleaner(detector, output_size)
        self._logger = logger or logging.getLogger(__name__)

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
