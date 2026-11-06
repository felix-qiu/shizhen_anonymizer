from pathlib import Path

import cv2
import numpy as np
import pytest

from src.errors import CleanerError, ErrorCode
from src.video.frame_extractor import FrameExtractor


def create_test_video(path: Path, frame_count: int = 5, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        pytest.skip("mp4v encoder is unavailable in this OpenCV build")
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_frame_extractor_reports_metadata_and_streams_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    create_test_video(video_path)

    with FrameExtractor(video_path) as extractor:
        frames = list(extractor.frames())
        assert extractor.width == 32
        assert extractor.height == 24
        assert extractor.fps == pytest.approx(10.0)
        assert extractor.total_frames == 5

    assert [frame.frame_index for frame in frames] == list(range(5))
    assert frames[2].timestamp == pytest.approx(0.2, abs=0.02)
    assert frames[0].image.shape == (24, 32, 3)


def test_missing_video_has_stable_error_code(tmp_path: Path) -> None:
    with pytest.raises(CleanerError) as error:
        FrameExtractor(tmp_path / "missing.mp4")

    assert error.value.code is ErrorCode.VIDEO_NOT_FOUND


def test_invalid_video_has_stable_error_code(tmp_path: Path) -> None:
    video_path = tmp_path / "invalid.mp4"
    video_path.write_text("not a video", encoding="utf-8")

    with pytest.raises(CleanerError) as error:
        FrameExtractor(video_path)

    assert error.value.code is ErrorCode.VIDEO_OPEN_FAILED


def test_abnormal_fps_uses_ffprobe_frame_count_and_average_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "input.avi"
    video_path.write_bytes(b"video")

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 960,
                cv2.CAP_PROP_FPS: 1000,
                cv2.CAP_PROP_FRAME_COUNT: 4962,
            }.get(property_id, 0)

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(
        FrameExtractor,
        "_probe_video_timing",
        staticmethod(lambda _path: (33.45, 166)),
    )

    with FrameExtractor(video_path) as extractor:
        assert extractor.fps == pytest.approx(33.45)
        assert extractor.total_frames == 166


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(0.0, False), (-1.0, False), (30.0, True), (120.0, True), (120.01, False)],
)
def test_reasonable_fps_range(fps: float, expected: bool) -> None:
    assert FrameExtractor._is_reasonable_fps(fps) is expected
