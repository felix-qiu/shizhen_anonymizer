"""Run the Phase 1 image cleaning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.detector.yolo_detector import YOLO11Detector
from src.errors import CleanerError
from src.image_io import load_image, save_image
from src.pipeline import ImageCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and crop an ultrasound ROI.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "input/test.png")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output/test_clean.png"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/config.yaml"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config, PROJECT_ROOT)
        image = load_image(args.input)
        detector = YOLO11Detector(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            device=config.device,
        )
        result = ImageCleaner(detector, config.output_size).clean(image)
        save_image(args.output, result.image)
    except CleanerError as exc:
        print(exc.code.value, file=sys.stderr)
        return 1

    print(f"Input: {args.input}")
    print(f"ROI confidence: {result.roi.confidence:.4f}")
    print(f"ROI bbox: {result.roi.bbox}")
    print(f"Inference time: {result.inference_time_ms:.2f} ms")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
