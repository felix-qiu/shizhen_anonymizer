import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.config import EvaluationConfig
from evaluation.evaluate_roi import evaluate_dataset
from evaluation.metrics import (
    DetectionSample,
    bbox_iou,
    calculate_metrics,
    roi_coverage,
)
from evaluation.report import generate_report
from src.dataset.yolo_dataset import read_annotations, scan_dataset, validate_dataset
from src.detector.roi_detector import ROIResult
from tools.dataset_statistics import calculate_statistics


def create_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def add_sample(
    root: Path,
    split: str,
    name: str,
    label: str = "0 0.5 0.5 0.8 0.6\n",
) -> Path:
    image_path = root / "images" / split / f"{name}.png"
    assert cv2.imwrite(str(image_path), np.full((100, 200, 3), 100, dtype=np.uint8))
    (root / "labels" / split / f"{name}.txt").write_text(label, encoding="utf-8")
    return image_path


def test_dataset_reader_loads_annotation_and_metadata(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    add_sample(tmp_path, "test", "001")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            [{"image": "001.png", "manufacturer": "CHISON", "model": "SonoEye"}]
        ),
        encoding="utf-8",
    )

    records = scan_dataset(tmp_path)
    annotation = read_annotations(records[0].label_path)[0]

    assert len(records) == 1
    assert records[0].manufacturer == "CHISON"
    assert records[0].model == "SonoEye"
    assert annotation.area_ratio == pytest.approx(0.48)
    assert annotation.to_xyxy(200, 100) == pytest.approx([20, 20, 180, 80])


def test_dataset_validation_detects_missing_empty_and_invalid_labels(
    tmp_path: Path,
) -> None:
    create_dataset(tmp_path)
    missing_image = tmp_path / "images/train/missing.png"
    assert cv2.imwrite(str(missing_image), np.zeros((10, 10, 3), dtype=np.uint8))
    add_sample(tmp_path, "val", "empty", label="")
    add_sample(tmp_path, "test", "invalid", label="0 0.9 0.5 0.4 0.5\n")

    errors = {issue.error for issue in validate_dataset(tmp_path)}

    assert "EMPTY_SPLIT" not in errors
    assert "LABEL_NOT_FOUND" in errors
    assert "EMPTY_ANNOTATION" in errors
    assert "BBOX_OUT_OF_RANGE" in errors


def test_dataset_statistics_include_resolution_device_and_roi_ratio(
    tmp_path: Path,
) -> None:
    create_dataset(tmp_path)
    add_sample(tmp_path, "test", "001")
    (tmp_path / "metadata.json").write_text(
        json.dumps([{"image": "001.png", "manufacturer": "GE", "model": "LOGIQ"}]),
        encoding="utf-8",
    )

    statistics = calculate_statistics(tmp_path)

    assert statistics["images"] == 1
    assert statistics["resolution"] == {"200x100": 1}
    assert statistics["manufacturers"] == {"GE": 1}
    assert statistics["models"] == {"LOGIQ": 1}
    assert statistics["annotations"] == 1
    assert statistics["roi_area_ratio"]["mean"] == pytest.approx(0.48)


def test_bbox_iou_and_roi_coverage() -> None:
    first = [0.0, 0.0, 10.0, 10.0]
    second = [5.0, 5.0, 15.0, 15.0]

    assert bbox_iou(first, first) == 1.0
    assert bbox_iou(first, second) == pytest.approx(25 / 175)
    assert roi_coverage(first, second) == 0.25
    assert roi_coverage(first, None) == 0.0


def test_metrics_include_detection_business_and_error_metrics() -> None:
    samples = [
        DetectionSample("ok.png", 100, 100, [10, 10, 90, 90], [10, 10, 90, 90], 0.99),
        DetectionSample("fn.png", 100, 100, [10, 10, 90, 90], None, None),
        DetectionSample("small.png", 100, 100, [10, 10, 90, 90], [10, 10, 20, 20], 0.4),
        DetectionSample("range.png", 100, 100, [10, 10, 90, 90], [-5, 0, 90, 90], 0.8),
    ]

    metrics = calculate_metrics(samples)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 0.5
    assert 0 <= metrics["mAP50_95"] <= metrics["mAP50"] <= 1
    assert metrics["errors"]["FN"] == ["fn.png"]
    assert "small.png" in metrics["errors"]["LOW_IOU"]
    assert "small.png" in metrics["errors"]["SMALL_ROI"]
    assert "range.png" in metrics["errors"]["OUT_OF_RANGE"]
    assert metrics["errors"]["LOW_CONFIDENCE"] == ["small.png"]


def test_evaluation_generates_metrics_visuals_errors_and_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "reports"
    create_dataset(dataset_root)
    add_sample(dataset_root, "test", "001")
    (dataset_root / "metadata.json").write_text(
        json.dumps([{"image": "001.png", "manufacturer": "Philips", "model": "EPIQ"}]),
        encoding="utf-8",
    )
    config = EvaluationConfig(0.5, 0.95, 0.7, 0.2, 0.001, output_dir)

    payload = evaluate_dataset(
        dataset_root,
        lambda image_path, image: ROIResult(0.98, [20, 20, 180, 80]),
        "test-model-v1",
        config,
    )

    assert payload["metrics"]["mAP50"] == 1.0
    assert payload["metrics"]["mAP50_95"] == 1.0
    assert payload["by_manufacturer"]["Philips"]["recall"] == 1.0
    assert payload["by_device_model"]["EPIQ"]["recall"] == 1.0
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "images/001.png").is_file()


def test_report_generation_contains_required_sections(tmp_path: Path) -> None:
    metrics = calculate_metrics(
        [DetectionSample("001.png", 100, 100, [0, 0, 100, 100], None, None)]
    )
    payload = {
        "model": "v1",
        "dataset": {
            "root": "/dataset",
            "test_images": 1,
            "manufacturers": {"GE": 1},
            "models": {"LOGIQ": 1},
        },
        "metrics": metrics,
        "by_manufacturer": {"GE": metrics},
        "by_device_model": {"LOGIQ": metrics},
    }

    report_path = generate_report(payload, tmp_path / "report.md")
    report = report_path.read_text(encoding="utf-8")

    assert "## Dataset" in report
    assert "## Error Analysis" in report
    assert "## Manufacturer Breakdown" in report
    assert "## Device Model Breakdown" in report
    assert "## Recommendations" in report
