from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detector.roi_detector import ROIDetector, ROIResult
from src.errors import CleanerError, ErrorCode
from src.video.video_processor import VideoProcessor


class CountingDetector(ROIDetector):
    def __init__(self, results: list[ROIResult | None] | None = None) -> None:
        self.calls = 0
        self.results = results or [ROIResult(0.95, [4, 2, 28, 22])]

    def detect(self, image: np.ndarray) -> ROIResult | None:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def create_test_video(path: Path, frame_count: int, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        pytest.skip("mp4v encoder is unavailable in this OpenCV build")
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()


def test_video_processor_reuses_roi_and_detects_at_interval(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    create_test_video(input_path, frame_count=65)
    detector = CountingDetector()

    result = VideoProcessor(detector, detect_interval=30, output_size=(48, 40)).process(
        input_path, output_path
    )

    assert result.processed_frames == 65
    assert result.detection_count == 3
    assert detector.calls == 3
    assert result.output_fps == pytest.approx(10.0)
    capture = cv2.VideoCapture(str(output_path))
    assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 65
    assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 48
    assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 40
    capture.release()


def test_redetection_miss_reuses_last_valid_roi(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    create_test_video(input_path, frame_count=4)
    detector = CountingDetector([ROIResult(0.9, [2, 2, 30, 22]), None])

    result = VideoProcessor(detector, detect_interval=2).process(
        input_path, output_path
    )

    assert result.processed_frames == 4
    assert detector.calls == 2


def test_first_frame_without_roi_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    create_test_video(input_path, frame_count=2)

    with pytest.raises(CleanerError) as error:
        VideoProcessor(CountingDetector([None])).process(
            input_path, tmp_path / "output.mp4"
        )

    assert error.value.code is ErrorCode.ROI_NOT_FOUND


def test_frame_crop_failure_has_stable_error_code(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    create_test_video(input_path, frame_count=2)
    invalid_roi = ROIResult(0.9, [100, 100, 200, 200])

    with pytest.raises(CleanerError) as error:
        VideoProcessor(CountingDetector([invalid_roi])).process(
            input_path, tmp_path / "output.mp4"
        )

    assert error.value.code is ErrorCode.FRAME_PROCESS_FAILED
