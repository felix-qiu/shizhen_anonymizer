"""Run the Phase 2 streaming video cleaning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.detector.yolo_detector import YOLO11Detector
from src.errors import CleanerError, ErrorCode
from src.video.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and crop ultrasound video ROI."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "input/test.mp4")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output/test_clean.mp4"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/config.yaml"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(ErrorCode.VIDEO_NOT_FOUND.value, file=sys.stderr)
        return 1
    try:
        config = load_config(args.config, PROJECT_ROOT)
        detector = YOLO11Detector(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            device=config.device,
        )
        processor = VideoProcessor(
            detector=detector,
            detect_interval=config.video.detect_interval,
            output_size=config.output_size,
            output_fps=config.video.output_fps,
        )
        result = processor.process(args.input, args.output)
    except CleanerError as exc:
        print(exc.code.value, file=sys.stderr)
        return 1

    print(f"Input video: {args.input}")
    print(f"FPS: {result.input_fps:.2f}")
    print(f"Frames: {result.source_frames}")
    print(f"ROI confidence: {result.initial_roi.confidence:.4f}")
    print(f"Processed frames: {result.processed_frames}")
    print(f"Output: {args.output}")
    print(f"Total time: {result.total_time_seconds:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
