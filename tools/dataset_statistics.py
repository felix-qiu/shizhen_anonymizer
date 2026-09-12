"""Generate image, annotation, device, and ROI-area dataset statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.yolo_dataset import scan_dataset
from src.errors import CleanerError
from src.image_io import load_image


def calculate_statistics(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    records = scan_dataset(root, strict=False)
    resolutions: Counter[str] = Counter()
    manufacturers: Counter[str] = Counter()
    models: Counter[str] = Counter()
    split_images: Counter[str] = Counter(
        {split: 0 for split in ("train", "val", "test")}
    )
    area_ratios: list[float] = []
    annotation_count = 0
    invalid_annotation_count = 0
    unreadable_images: list[str] = []

    for record in records:
        split_images[record.split] += 1
        try:
            image = load_image(record.image_path)
            height, width = image.shape[:2]
            resolutions[f"{width}x{height}"] += 1
        except CleanerError:
            unreadable_images.append(str(record.image_path.relative_to(root)))
        if record.manufacturer:
            manufacturers[record.manufacturer] += 1
        if record.model:
            models[record.model] += 1
        annotation_count += len(record.annotations)
        for annotation in record.annotations:
            if annotation.is_valid():
                area_ratios.append(annotation.area_ratio)
            else:
                invalid_annotation_count += 1

    ratio_summary = {
        "mean": fmean(area_ratios) if area_ratios else None,
        "min": min(area_ratios) if area_ratios else None,
        "max": max(area_ratios) if area_ratios else None,
    }
    return {
        "images": len(records),
        "images_by_split": dict(sorted(split_images.items())),
        "resolution": dict(sorted(resolutions.items())),
        "manufacturers": dict(sorted(manufacturers.items())),
        "models": dict(sorted(models.items())),
        "annotations": annotation_count,
        "invalid_annotations": invalid_annotation_count,
        "roi_area_ratio": ratio_summary,
        "unreadable_images": unreadable_images,
    }


def write_statistics(statistics: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the ultrasound ROI dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/ultrasound_roi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/dataset_statistics.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    statistics = calculate_statistics(args.dataset)
    output = write_statistics(statistics, args.output)
    print(f"Images: {statistics['images']}")
    print(f"Annotations: {statistics['annotations']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
