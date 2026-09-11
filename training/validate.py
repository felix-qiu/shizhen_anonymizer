"""Validate a trained YOLO11 ultrasound ROI detector."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    load_dataset_config,
    load_training_config,
    require_model_file,
    resolved_dataset_yaml,
)


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    map50: float
    map50_95: float
    precision: float
    recall: float


def _metric(results: Any, key: str, attribute: str) -> float:
    result_values = getattr(results, "results_dict", {})
    if key in result_values:
        return float(result_values[key])
    box = getattr(results, "box", None)
    if box is None or not hasattr(box, attribute):
        raise RuntimeError(f"validation result does not contain {key}")
    return float(getattr(box, attribute))


def validate_model(
    model_path: str | Path,
    dataset_yaml: str | Path,
    imgsz: int,
    yolo_factory: Callable[[str], Any] | None = None,
) -> ValidationMetrics:
    if imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    model_file = require_model_file(model_path)
    dataset = load_dataset_config(dataset_yaml, PROJECT_ROOT)
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO

    with resolved_dataset_yaml(dataset) as data_yaml:
        results = yolo_factory(str(model_file)).val(data=str(data_yaml), imgsz=imgsz)
    return ValidationMetrics(
        map50=_metric(results, "metrics/mAP50(B)", "map50"),
        map50_95=_metric(results, "metrics/mAP50-95(B)", "map"),
        precision=_metric(results, "metrics/precision(B)", "mp"),
        recall=_metric(results, "metrics/recall(B)", "mr"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ultrasound ROI model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/train.yaml"
    )
    parser.add_argument("--imgsz", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config, PROJECT_ROOT)
    metrics = validate_model(
        args.model,
        (args.data or config.dataset_yaml).resolve(),
        args.imgsz if args.imgsz is not None else config.imgsz,
    )
    print(f"mAP50: {metrics.map50:.6f}")
    print(f"mAP50-95: {metrics.map50_95:.6f}")
    print(f"Precision: {metrics.precision:.6f}")
    print(f"Recall: {metrics.recall:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
