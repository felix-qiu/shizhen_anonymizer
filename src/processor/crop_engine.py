"""Safe ROI cropping and aspect-ratio-preserving standardization."""

from collections.abc import Sequence

import cv2
import numpy as np

OutputSize = int | tuple[int, int]


def _normalize_output_size(output_size: OutputSize) -> tuple[int, int]:
    if isinstance(output_size, int):
        width = height = output_size
    else:
        width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size dimensions must be positive")
    return width, height


def _letterbox(image: np.ndarray, output_size: OutputSize) -> np.ndarray:
    target_width, target_height = _normalize_output_size(output_size)
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )

    if image.ndim == 2:
        canvas = np.zeros((target_height, target_width), dtype=image.dtype)
    else:
        canvas = np.zeros(
            (target_height, target_width, image.shape[2]), dtype=image.dtype
        )
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height, x_offset : x_offset + resized_width
    ] = resized
    return canvas


def crop_image(
    image: np.ndarray,
    bbox: Sequence[int | float],
    output_size: OutputSize | None = None,
) -> np.ndarray:
    """Crop a clamped bbox and optionally letterbox it to a standard size."""

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = (round(float(value)) for value in bbox)
    x1 = min(max(x1, 0), image_width)
    x2 = min(max(x2, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    y2 = min(max(y2, 0), image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox is empty after clamping to image bounds")

    cropped = image[y1:y2, x1:x2].copy()
    return _letterbox(cropped, output_size) if output_size is not None else cropped
