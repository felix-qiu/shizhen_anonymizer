"""Detection and ultrasound ROI business metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

BBox = list[float]


@dataclass(frozen=True, slots=True)
class DetectionSample:
    image: str
    image_width: int
    image_height: int
    gt_bbox: BBox
    pred_bbox: BBox | None
    confidence: float | None
    manufacturer: str | None = None
    model: str | None = None


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first: BBox, second: BBox) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def bbox_iou(first: BBox, second: BBox) -> float:
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def roi_coverage(gt_bbox: BBox, pred_bbox: BBox | None) -> float:
    if pred_bbox is None:
        return 0.0
    gt_area = _area(gt_bbox)
    return _intersection_area(gt_bbox, pred_bbox) / gt_area if gt_area > 0 else 0.0


def bbox_is_in_bounds(box: BBox, image_width: int, image_height: int) -> bool:
    return (
        len(box) == 4
        and 0 <= box[0] < box[2] <= image_width
        and 0 <= box[1] < box[3] <= image_height
    )


def average_precision(samples: list[DetectionSample], iou_threshold: float) -> float:
    if not samples:
        return 0.0
    predictions = sorted(
        (sample for sample in samples if sample.pred_bbox is not None),
        key=lambda sample: sample.confidence or 0.0,
        reverse=True,
    )
    if not predictions:
        return 0.0

    true_positives = 0
    false_positives = 0
    precision_points: list[float] = []
    recall_points: list[float] = []
    for sample in predictions:
        pred_bbox = sample.pred_bbox
        if (
            pred_bbox is not None
            and bbox_iou(sample.gt_bbox, pred_bbox) >= iou_threshold
        ):
            true_positives += 1
        else:
            false_positives += 1
        precision_points.append(true_positives / (true_positives + false_positives))
        recall_points.append(true_positives / len(samples))

    interpolated: list[float] = []
    for step in range(101):
        recall_level = step / 100
        candidates = [
            precision
            for precision, recall in zip(precision_points, recall_points, strict=True)
            if recall >= recall_level
        ]
        interpolated.append(max(candidates, default=0.0))
    return sum(interpolated) / 101


def calculate_metrics(
    samples: list[DetectionSample],
    decision_iou: float = 0.5,
    coverage_threshold: float = 0.95,
    low_confidence_threshold: float = 0.7,
    small_roi_threshold: float = 0.2,
) -> dict[str, Any]:
    true_positives = sum(
        sample.pred_bbox is not None
        and bbox_iou(sample.gt_bbox, sample.pred_bbox) >= decision_iou
        for sample in samples
    )
    predictions = sum(sample.pred_bbox is not None for sample in samples)
    precision = true_positives / predictions if predictions else 0.0
    recall = true_positives / len(samples) if samples else 0.0

    thresholds = [0.5 + index * 0.05 for index in range(10)]
    ap_values = {
        f"{threshold:.2f}": average_precision(samples, threshold)
        for threshold in thresholds
    }
    coverage_values = [
        roi_coverage(sample.gt_bbox, sample.pred_bbox) for sample in samples
    ]

    errors: dict[str, list[str]] = {
        "FN": [],
        "LOW_IOU": [],
        "SMALL_ROI": [],
        "OUT_OF_RANGE": [],
        "LOW_CONFIDENCE": [],
    }
    for sample in samples:
        if sample.pred_bbox is None:
            errors["FN"].append(sample.image)
            continue
        if bbox_iou(sample.gt_bbox, sample.pred_bbox) < decision_iou:
            errors["LOW_IOU"].append(sample.image)
        image_area = sample.image_width * sample.image_height
        if (
            image_area > 0
            and _area(sample.pred_bbox) / image_area < small_roi_threshold
        ):
            errors["SMALL_ROI"].append(sample.image)
        if not bbox_is_in_bounds(
            sample.pred_bbox, sample.image_width, sample.image_height
        ):
            errors["OUT_OF_RANGE"].append(sample.image)
        if (sample.confidence or 0.0) < low_confidence_threshold:
            errors["LOW_CONFIDENCE"].append(sample.image)

    risk_samples = sorted(
        set(errors["FN"] + errors["LOW_IOU"] + errors["LOW_CONFIDENCE"])
    )
    coverage_mean = fmean(coverage_values) if coverage_values else 0.0
    coverage_pass_rate = (
        sum(value >= coverage_threshold for value in coverage_values)
        / len(coverage_values)
        if coverage_values
        else 0.0
    )
    return {
        "samples": len(samples),
        "precision": precision,
        "recall": recall,
        "mAP50": ap_values["0.50"],
        "mAP50_95": fmean(ap_values.values()) if ap_values else 0.0,
        "ap_by_iou": ap_values,
        "roi_coverage": {
            "mean": coverage_mean,
            "pass_rate": coverage_pass_rate,
            "threshold": coverage_threshold,
        },
        "sensitive_information_residual_risk": {
            "sample_count": len(risk_samples),
            "samples": risk_samples,
            "low_confidence_threshold": low_confidence_threshold,
            "low_confidence_samples": errors["LOW_CONFIDENCE"],
        },
        "errors": errors,
    }
