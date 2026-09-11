"""Ultralytics YOLO11 adapter for ultrasound ROI detection."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.detector.roi_detector import ROIDetector, ROIResult
from src.errors import CleanerError, ErrorCode


class YOLO11Detector(ROIDetector):
    """Detect the highest-confidence ``ultrasound_roi`` YOLO box."""

    TARGET_CLASS = "ultrasound_roi"

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        device: str = "auto",
        *,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)

        if model is None:
            if not self.model_path.is_file():
                raise CleanerError(ErrorCode.MODEL_NOT_FOUND, str(self.model_path))
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "ultralytics is not installed; run pip install -r requirements.txt"
                ) from exc
            model = YOLO(str(self.model_path))
        self._model = model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # Allows lightweight unit tests with an injected model.
            return "cpu"

    @staticmethod
    def _scalar(value: Any) -> float:
        return float(value.item() if hasattr(value, "item") else value)

    @staticmethod
    def _coordinates(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return [float(coordinate) for coordinate in value]

    @staticmethod
    def _class_name(names: dict[int, str] | list[str], class_id: int) -> str | None:
        if isinstance(names, dict):
            return names.get(class_id)
        return names[class_id] if 0 <= class_id < len(names) else None

    def detect(self, image: np.ndarray) -> ROIResult | None:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty OpenCV numpy array")

        predictions: Iterable[Any] = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        best: ROIResult | None = None
        for prediction in predictions:
            names = getattr(prediction, "names", None) or getattr(
                self._model, "names", {}
            )
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(self._scalar(box.cls))
                if self._class_name(names, class_id) != self.TARGET_CLASS:
                    continue
                confidence = self._scalar(box.conf)
                if confidence < self.confidence_threshold:
                    continue
                coordinates = self._coordinates(box.xyxy)
                if len(coordinates) != 4:
                    continue
                candidate = ROIResult(
                    confidence=confidence,
                    bbox=[round(coordinate) for coordinate in coordinates],
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        return best
