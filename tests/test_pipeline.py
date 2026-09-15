import numpy as np
import pytest

from src.detector.roi_detector import ROIDetector, ROIResult
from src.errors import CleanerError, ErrorCode
from src.pipeline import ImageCleaner


class StubDetector(ROIDetector):
    def __init__(self, result: ROIResult | None) -> None:
        self.result = result

    def detect(self, image: np.ndarray) -> ROIResult | None:
        return self.result


def test_roi_not_found_has_stable_error_code() -> None:
    cleaner = ImageCleaner(StubDetector(None), output_size=None)

    with pytest.raises(CleanerError) as error:
        cleaner.clean(np.zeros((10, 10, 3), dtype=np.uint8))

    assert error.value.code is ErrorCode.ROI_NOT_FOUND


def test_pipeline_only_depends_on_detector_interface() -> None:
    roi = ROIResult(confidence=0.98, bbox=[1, 2, 8, 9])
    cleaner = ImageCleaner(StubDetector(roi), output_size=None)

    result = cleaner.clean(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result.roi == ROIResult(confidence=0.98, bbox=[0, 2, 10, 10])
    assert result.image.shape == (8, 10, 3)
