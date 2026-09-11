from pathlib import Path

import cv2
import numpy as np
import pytest

from src.errors import CleanerError, ErrorCode
from src.image_io import load_image


def test_load_image(tmp_path: Path) -> None:
    expected = np.full((6, 7, 3), 123, dtype=np.uint8)
    image_path = tmp_path / "test.png"
    assert cv2.imwrite(str(image_path), expected)

    actual = load_image(image_path)

    np.testing.assert_array_equal(actual, expected)


def test_missing_image_has_stable_error_code(tmp_path: Path) -> None:
    with pytest.raises(CleanerError) as error:
        load_image(tmp_path / "missing.png")

    assert error.value.code is ErrorCode.IMAGE_NOT_FOUND


def test_invalid_image_has_stable_error_code(tmp_path: Path) -> None:
    image_path = tmp_path / "invalid.png"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(CleanerError) as error:
        load_image(image_path)

    assert error.value.code is ErrorCode.IMAGE_LOAD_FAILED
