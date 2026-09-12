"""Draw ground-truth and predicted ROI boxes for manual review."""

from pathlib import Path

import cv2
import numpy as np

from evaluation.metrics import BBox
from src.image_io import save_image


def _draw_box(
    image: np.ndarray,
    bbox: BBox,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = (round(value) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(18, y1 - 8)
    cv2.putText(
        image,
        label,
        (max(0, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def visualize_prediction(
    image: np.ndarray,
    gt_bbox: BBox,
    pred_bbox: BBox | None,
    output_path: str | Path,
    confidence: float | None = None,
) -> Path:
    canvas = image.copy()
    _draw_box(canvas, gt_bbox, (0, 255, 0), "GT")
    if pred_bbox is not None:
        label = "Prediction" if confidence is None else f"Prediction {confidence:.3f}"
        _draw_box(canvas, pred_bbox, (0, 0, 255), label)
    else:
        cv2.putText(
            canvas,
            "Prediction: NOT FOUND",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    destination = Path(output_path)
    save_image(destination, canvas)
    return destination
