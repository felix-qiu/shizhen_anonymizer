"""Image decoding and encoding with explicit domain errors."""

from pathlib import Path

import cv2
import numpy as np

from src.errors import CleanerError, ErrorCode


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise CleanerError(ErrorCode.IMAGE_NOT_FOUND, str(image_path))
    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error) as exc:
        raise CleanerError(ErrorCode.IMAGE_LOAD_FAILED, str(image_path)) from exc
    if image is None or image.size == 0:
        raise CleanerError(ErrorCode.IMAGE_LOAD_FAILED, str(image_path))
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix.lower()
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Unable to encode output image as {extension}")
    encoded.tofile(output_path)
