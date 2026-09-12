"""Run ROI prediction, metrics, visualization, and error analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import EvaluationConfig, load_evaluation_config
from evaluation.metrics import DetectionSample, calculate_metrics
from evaluation.report import generate_report
from evaluation.visualize import visualize_prediction
from src.dataset.yolo_dataset import DatasetRecord, scan_dataset, validate_dataset
from src.detector.roi_detector import ROIResult
from src.detector.yolo_detector import YOLO11Detector
from src.image_io import load_image

PredictionProvider = Callable[[Path, np.ndarray], ROIResult | None]


def load_predictions(path: str | Path) -> dict[str, ROIResult | None]:
    prediction_path = Path(path)
    raw: Any = json.loads(prediction_path.read_text(encoding="utf-8"))
    entries = raw.get("predictions", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise TypeError("predictions JSON must be a list or contain a predictions list")
    predictions: dict[str, ROIResult | None] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("image"):
            raise ValueError("each prediction must contain an image field")
        key = Path(str(entry["image"])).name
        bbox = entry.get("bbox")
        if bbox is None:
            predictions[key] = None
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"prediction bbox must contain four values: {key}")
        predictions[key] = ROIResult(
            confidence=float(entry.get("confidence", 0.0)),
            bbox=[round(float(value)) for value in bbox],
        )
    return predictions


def prediction_file_provider(path: str | Path) -> PredictionProvider:
    predictions = load_predictions(path)

    def provide(image_path: Path, image: np.ndarray) -> ROIResult | None:
        return predictions.get(image_path.name)

    return provide


def model_provider(
    model_path: str | Path, confidence: float, device: str
) -> PredictionProvider:
    detector = YOLO11Detector(model_path, confidence, device)

    def provide(image_path: Path, image: np.ndarray) -> ROIResult | None:
        return detector.detect(image)

    return provide


def _validate_test_records(records: list[DatasetRecord]) -> None:
    if not records:
        raise ValueError("test dataset contains no supported images")
    for record in records:
        if len(record.annotations) != 1 or not record.annotations[0].is_valid():
            raise ValueError(
                f"test image must have exactly one valid ultrasound ROI: {record.image_path}"
            )


def _manufacturer_metrics(
    samples: list[DetectionSample], config: EvaluationConfig
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[DetectionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.manufacturer or "UNKNOWN", []).append(sample)
    return {
        manufacturer: calculate_metrics(
            group,
            config.decision_iou,
            config.coverage_threshold,
            config.low_confidence_threshold,
            config.small_roi_threshold,
        )
        for manufacturer, group in grouped.items()
    }


def _device_model_metrics(
    samples: list[DetectionSample], config: EvaluationConfig
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[DetectionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.model or "UNKNOWN", []).append(sample)
    return {
        model: calculate_metrics(
            group,
            config.decision_iou,
            config.coverage_threshold,
            config.low_confidence_threshold,
            config.small_roi_threshold,
        )
        for model, group in grouped.items()
    }


def evaluate_dataset(
    dataset_root: str | Path,
    provider: PredictionProvider,
    model_name: str,
    config: EvaluationConfig,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    issues = [issue for issue in validate_dataset(root) if issue.split == "test"]
    if issues:
        raise ValueError(f"test dataset has {len(issues)} validation errors")
    records = [record for record in scan_dataset(root) if record.split == "test"]
    _validate_test_records(records)

    images_dir = config.output_dir / "images"
    errors_dir = config.output_dir / "errors"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old_visualization in images_dir.glob("*.png"):
        old_visualization.unlink()
    samples: list[DetectionSample] = []
    visualizations: dict[str, Path] = {}

    for record in records:
        image = load_image(record.image_path)
        height, width = image.shape[:2]
        gt_bbox = record.annotations[0].to_xyxy(width, height)
        prediction = provider(record.image_path, image)
        pred_bbox = (
            [float(value) for value in prediction.bbox]
            if prediction is not None
            else None
        )
        sample = DetectionSample(
            image=record.image_path.name,
            image_width=width,
            image_height=height,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            confidence=prediction.confidence if prediction is not None else None,
            manufacturer=record.manufacturer,
            model=record.model,
        )
        samples.append(sample)
        visualization = images_dir / f"{record.image_path.stem}.png"
        visualize_prediction(
            image, gt_bbox, pred_bbox, visualization, sample.confidence
        )
        visualizations[sample.image] = visualization

    metrics = calculate_metrics(
        samples,
        config.decision_iou,
        config.coverage_threshold,
        config.low_confidence_threshold,
        config.small_roi_threshold,
    )
    for category, image_names in metrics["errors"].items():
        category_dir = errors_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for old_error in category_dir.glob("*.png"):
            old_error.unlink()
        for image_name in image_names:
            source = visualizations[image_name]
            shutil.copy2(source, category_dir / source.name)

    manufacturers = Counter(
        sample.manufacturer for sample in samples if sample.manufacturer
    )
    models = Counter(sample.model for sample in samples if sample.model)
    payload = {
        "model": model_name,
        "dataset": {
            "root": str(root),
            "test_images": len(samples),
            "manufacturers": dict(sorted(manufacturers.items())),
            "models": dict(sorted(models.items())),
        },
        "thresholds": {
            "decision_iou": config.decision_iou,
            "coverage": config.coverage_threshold,
            "low_confidence": config.low_confidence_threshold,
            "small_roi": config.small_roi_threshold,
        },
        "metrics": metrics,
        "by_manufacturer": _manufacturer_metrics(samples, config),
        "by_device_model": _device_model_metrics(samples, config),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generate_report(payload, config.output_dir / "report.md")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ultrasound ROI predictions.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/ultrasound_roi",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/evaluation.yaml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model is not None and args.predictions is not None:
        raise ValueError("use either --model or --predictions, not both")
    config = load_evaluation_config(args.config, PROJECT_ROOT)
    if args.output_dir is not None:
        config = EvaluationConfig(
            config.decision_iou,
            config.coverage_threshold,
            config.low_confidence_threshold,
            config.small_roi_threshold,
            config.inference_confidence,
            args.output_dir.resolve(),
        )

    if args.predictions is not None:
        provider = prediction_file_provider(args.predictions)
        model_name = f"predictions:{args.predictions}"
    else:
        model_path = args.model or PROJECT_ROOT / "models/yolo11s_roi.pt"
        provider = model_provider(model_path, config.inference_confidence, args.device)
        model_name = str(model_path)
    payload = evaluate_dataset(args.dataset, provider, model_name, config)
    metrics = payload["metrics"]
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"mAP50: {metrics['mAP50']:.6f}")
    print(f"mAP50-95: {metrics['mAP50_95']:.6f}")
    print(f"Report: {config.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
