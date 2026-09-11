"""Export or package a trained ultrasound ROI model."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    load_training_config,
    require_model_file,
    validate_model_version,
)

ExportFormat = Literal["pt", "onnx"]


def export_model(
    model_path: str | Path,
    export_format: ExportFormat,
    imgsz: int,
    output_dir: str | Path | None = None,
    yolo_factory: Callable[[str], Any] | None = None,
    artifact_name: str | None = None,
) -> Path:
    """Export ONNX, or copy a PT checkpoint into the selected artifact directory."""

    if imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if artifact_name is not None and Path(artifact_name).name != artifact_name:
        raise ValueError("artifact_name must be a file name without directories")
    model_file = require_model_file(model_path)
    destination = Path(output_dir).resolve() if output_dir else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    if export_format == "pt":
        if destination is None:
            return model_file
        target = destination / (artifact_name or model_file.name)
        if target != model_file:
            shutil.copy2(model_file, target)
        return target
    if export_format != "onnx":
        raise ValueError(f"unsupported export format: {export_format}")

    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    exported = Path(
        yolo_factory(str(model_file)).export(format="onnx", imgsz=imgsz)
    ).resolve()
    if destination is None:
        return exported
    target = destination / (artifact_name or exported.name)
    if target != exported:
        shutil.copy2(exported, target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an ultrasound ROI model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", choices=("pt", "onnx"), default="onnx")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/train.yaml"
    )
    parser.add_argument("--imgsz", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config, PROJECT_ROOT)
    version = validate_model_version(args.version or config.model_version)
    artifact_name = f"yolo11s_ultrasound_roi_{version}.{args.format}"
    exported = export_model(
        args.model,
        args.format,
        args.imgsz if args.imgsz is not None else config.imgsz,
        args.output_dir or config.export_dir,
        artifact_name=artifact_name,
    )
    print(f"Exported model: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
