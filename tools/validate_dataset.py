"""Validate image-label pairing and normalized ROI bounding boxes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.yolo_dataset import DatasetValidationIssue, validate_dataset


def write_validation_result(
    issues: list[DatasetValidationIssue], output_path: str | Path
) -> Path:
    payload = {
        "valid": not issues,
        "error_count": len(issues),
        "errors": [asdict(issue) for issue in issues],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ultrasound ROI dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/ultrasound_roi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/invalid_samples.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues = validate_dataset(args.dataset)
    output = write_validation_result(issues, args.output)
    print(f"Errors: {len(issues)}")
    print(f"Output: {output}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
