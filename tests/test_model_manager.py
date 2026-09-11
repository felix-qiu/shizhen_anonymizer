from typing import Any

import numpy as np

from src.detector.roi_detector import ROIResult
from src.model import model_manager as model_manager_module
from src.model.model_manager import ModelManager


class FakeDetector:
    instances = 0

    def __init__(self, **kwargs: Any) -> None:
        FakeDetector.instances += 1

    def detect(self, image: np.ndarray) -> ROIResult:
        return ROIResult(0.9, [0, 0, 1, 1])


def test_model_manager_loads_detector_once(monkeypatch) -> None:
    FakeDetector.instances = 0
    monkeypatch.setattr(model_manager_module, "YOLO11Detector", FakeDetector)
    manager = ModelManager("model.pt", 0.7, "cpu")

    first = manager.get_model()
    second = manager.get_model()

    assert first is second
    assert FakeDetector.instances == 1
    assert first.detect(np.zeros((1, 1, 3), dtype=np.uint8)).confidence == 0.9
