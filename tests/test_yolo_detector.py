from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from src.detector.yolo_detector import YOLO11Detector
from src.errors import CleanerError, ErrorCode


class FakeModel:
    names: ClassVar[dict[int, str]] = {0: "other", 1: "ultrasound_roi"}

    def __init__(self, boxes: list[SimpleNamespace]) -> None:
        self.boxes = boxes

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(boxes=self.boxes, names=self.names)]


def test_missing_model_has_stable_error_code(tmp_path: Path) -> None:
    with pytest.raises(CleanerError) as error:
        YOLO11Detector(tmp_path / "missing.pt", 0.7, device="cpu")

    assert error.value.code is ErrorCode.MODEL_NOT_FOUND


def test_detector_returns_none_when_roi_does_not_exist() -> None:
    model = FakeModel(
        [
            SimpleNamespace(
                cls=np.array([0]), conf=np.array([0.99]), xyxy=np.array([[1, 2, 8, 9]])
            )
        ]
    )
    detector = YOLO11Detector("unused.pt", 0.7, device="cpu", model=model)

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_detector_selects_highest_confidence_ultrasound_roi() -> None:
    boxes = [
        SimpleNamespace(
            cls=np.array([1]),
            conf=np.array([0.75]),
            xyxy=np.array([[1.2, 2.1, 8.2, 9.1]]),
        ),
        SimpleNamespace(
            cls=np.array([1]),
            conf=np.array([0.95]),
            xyxy=np.array([[2.2, 3.1, 9.2, 10.1]]),
        ),
    ]
    detector = YOLO11Detector("unused.pt", 0.7, device="cpu", model=FakeModel(boxes))

    result = detector.detect(np.zeros((12, 12, 3), dtype=np.uint8))

    assert result is not None
    assert result.confidence == 0.95
    assert result.bbox == [2, 3, 9, 10]
