import numpy as np
import pytest

from src.processor.crop_engine import crop_image, top_boundary_bbox


def test_crop_image_uses_bbox() -> None:
    image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)

    cropped = crop_image(image, [2, 3, 9, 8])

    assert cropped.shape == (5, 7, 3)
    np.testing.assert_array_equal(cropped, image[3:8, 2:9])


def test_crop_image_clamps_out_of_bounds_coordinates() -> None:
    image = np.ones((8, 10, 3), dtype=np.uint8)

    cropped = crop_image(image, [-100, -20, 100, 50])

    assert cropped.shape == image.shape


def test_resize_letterboxes_without_distorting_aspect_ratio() -> None:
    image = np.full((4, 8, 3), 255, dtype=np.uint8)

    resized = crop_image(image, [0, 0, 8, 4], output_size=16)

    assert resized.shape == (16, 16, 3)
    assert np.all(resized[4:12] == 255)
    assert np.all(resized[:4] == 0)
    assert np.all(resized[12:] == 0)


def test_empty_bbox_is_rejected() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty after clamping"):
        crop_image(image, [20, 20, 30, 30])


def test_top_boundary_bbox_preserves_left_right_and_bottom_edges() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    assert top_boundary_bbox(image, [2, 3, 28, 18]) == [0, 3, 30, 20]
