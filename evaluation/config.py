"""Validated ROI evaluation settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    decision_iou: float
    coverage_threshold: float
    low_confidence_threshold: float
    small_roi_threshold: float
    inference_confidence: float
    output_dir: Path


def load_evaluation_config(
    config_path: str | Path, project_root: str | Path
) -> EvaluationConfig:
    path = Path(config_path)
    project = Path(project_root).resolve()
    if not path.is_absolute():
        path = project / path
    with path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}
    evaluation = raw["evaluation"]
    values = [
        float(evaluation[name])
        for name in (
            "decision_iou",
            "coverage_threshold",
            "low_confidence_threshold",
            "small_roi_threshold",
            "inference_confidence",
        )
    ]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("evaluation thresholds must be between 0 and 1")
    output_dir = Path(raw["output"]["dir"])
    if not output_dir.is_absolute():
        output_dir = project / output_dir
    return EvaluationConfig(*values, output_dir.resolve())
