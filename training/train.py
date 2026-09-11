"""Train a YOLO11 ultrasound ROI detector."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    TrainingConfig,
    load_dataset_config,
    load_training_config,
    resolved_dataset_yaml,
)


def train_model(
    config: TrainingConfig, yolo_factory: Callable[[str], Any] | None = None
) -> Path:
    """Train with resolved settings and return the expected best-weight path."""

    dataset = load_dataset_config(config.dataset_yaml, PROJECT_ROOT)
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO

    model = yolo_factory(config.model)
    with resolved_dataset_yaml(dataset) as data_yaml:
        results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            project=str(config.output_dir),
            name=config.run_name,
            save=True,
        )
    save_dir = Path(getattr(results, "save_dir", config.output_dir / config.run_name))
    return save_dir / "weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ultrasound ROI model.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/train.yaml"
    )
    parser.add_argument("--model")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_training_config(args.config, PROJECT_ROOT)
    dataset_yaml = (args.data or base.dataset_yaml).resolve()
    dataset = load_dataset_config(dataset_yaml, PROJECT_ROOT)
    config = TrainingConfig(
        model=args.model or base.model,
        model_version=base.model_version,
        dataset_yaml=dataset_yaml,
        dataset_root=dataset.root,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        imgsz=args.imgsz if args.imgsz is not None else base.imgsz,
        batch=args.batch if args.batch is not None else base.batch,
        output_dir=(args.output_dir or base.output_dir).resolve(),
        run_name=args.name or base.run_name,
        export_dir=base.export_dir,
    )
    if min(config.epochs, config.imgsz, config.batch) <= 0:
        raise ValueError("epochs, imgsz, and batch must be positive integers")
    best_model = train_model(config)
    print(f"Best model: {best_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
