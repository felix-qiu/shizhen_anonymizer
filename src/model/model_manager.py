"""Thread-safe singleton lifecycle for the shared YOLO detector."""

from threading import Lock
from typing import Any

from src.detector.roi_detector import ROIDetector, ROIResult
from src.detector.yolo_detector import YOLO11Detector


class _SynchronizedDetector(ROIDetector):
    def __init__(self, detector: ROIDetector) -> None:
        self._detector = detector
        self._inference_lock = Lock()

    def detect(self, image: Any) -> ROIResult | None:
        with self._inference_lock:
            return self._detector.detect(image)


class ModelManager:
    """Load one detector per application process and reuse it for all requests."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model: ROIDetector | None = None
        self._lifecycle_lock = Lock()

    def load(self) -> ROIDetector:
        if self._model is not None:
            return self._model
        with self._lifecycle_lock:
            if self._model is None:
                detector = YOLO11Detector(
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                )
                self._model = _SynchronizedDetector(detector)
        return self._model

    def get_model(self) -> ROIDetector:
        return self.load()

    def close(self) -> None:
        with self._lifecycle_lock:
            self._model = None
