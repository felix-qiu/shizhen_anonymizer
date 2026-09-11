from pathlib import Path

import cv2
import numpy as np
import pytest

from src.errors import CleanerError, ErrorCode
from src.video.video_writer import VideoWriter


def test_video_writer_creates_readable_mp4(tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"
    try:
        with VideoWriter(output_path, fps=12.0, size=(40, 30)) as writer:
            for index in range(4):
                writer.write(np.full((30, 40, 3), index * 30, dtype=np.uint8))
    except CleanerError as exc:
        if exc.code is ErrorCode.VIDEO_WRITE_FAILED:
            pytest.skip("mp4v encoder is unavailable in this OpenCV build")
        raise

    capture = cv2.VideoCapture(str(output_path))
    assert capture.isOpened()
    assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    capture.release()


def test_video_writer_rejects_non_mp4_output(tmp_path: Path) -> None:
    with pytest.raises(CleanerError) as error:
        VideoWriter(tmp_path / "output.avi", fps=12.0, size=(40, 30))

    assert error.value.code is ErrorCode.VIDEO_WRITE_FAILED
