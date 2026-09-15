"""Merge flat AnyLabeling exports into a grouped top-boundary YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass(frozen=True, slots=True)
class Sample:
    image: Path
    label: Path
    digest: str
    top: float
    group: str
    source: str


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _group_key(stem: str) -> str:
    """Derive a conservative study/sequence key from exported filenames."""

    name = re.sub(r"^\d+_", "", stem)
    dicom = re.match(r"(.+?)\.(\d{8})\d{6}\.\d+$", name)
    if dicom:
        return f"dicom:{dicom.group(1)}:{dicom.group(2)}"
    patient = re.match(r"(Patient_\d{14})", name, re.IGNORECASE)
    if patient:
        return patient.group(1).lower()
    usimage = re.match(r"(usimage\d{8}\d{4})", name, re.IGNORECASE)
    if usimage:
        return usimage.group(1).lower()
    pneumonia = re.match(r"(PNEUMONIE\d*)_", name, re.IGNORECASE)
    if pneumonia:
        return pneumonia.group(1).lower()
    # Remove common frame/image suffixes while retaining the case identifier.
    return re.sub(r"(?:_\d+){1,2}$", "", name).lower()


def _read_top(label: Path) -> float:
    lines = [
        line.strip()
        for line in label.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one bbox: {label} ({len(lines)} found)")
    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"invalid YOLO label: {label}")
    class_id = int(fields[0])
    x_center, y_center, width, height = map(float, fields[1:])
    if class_id != 0:
        raise ValueError(f"unexpected class {class_id}: {label}")
    if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
        raise ValueError(f"bbox coordinate outside [0, 1]: {label}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"empty bbox: {label}")
    return min(max(y_center - height / 2.0, 0.0), 1.0)


def collect_samples(sources: list[Path]) -> tuple[list[Sample], list[dict[str, str]]]:
    samples: list[Sample] = []
    duplicates: list[dict[str, str]] = []
    digest_owner: dict[str, Path] = {}
    for source in sources:
        image_dir = source / "images"
        label_dir = source / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"AnyLabeling images/labels folders not found: {source}")
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"label not found for {image}")
            digest = _digest(image)
            if digest in digest_owner:
                duplicates.append(
                    {"removed": str(image), "kept": str(digest_owner[digest])}
                )
                continue
            digest_owner[digest] = image
            samples.append(
                Sample(
                    image=image,
                    label=label,
                    digest=digest,
                    top=_read_top(label),
                    group=_group_key(image.stem),
                    source=source.name,
                )
            )
    return samples, duplicates


def grouped_split(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)

    rng = random.Random(seed)
    ordered = list(groups.items())
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)
    targets = {name: len(samples) * ratio for name, ratio in SPLIT_RATIOS.items()}
    result: dict[str, list[Sample]] = {name: [] for name in SPLIT_RATIOS}
    for _, group_samples in ordered:
        split = max(
            SPLIT_RATIOS,
            key=lambda name: (targets[name] - len(result[name])) / targets[name],
        )
        result[split].extend(group_samples)
    return result


def _unique_name(sample: Sample, occupied: set[str]) -> str:
    name = sample.image.name
    if name.lower() not in occupied:
        occupied.add(name.lower())
        return name
    name = f"{sample.source}_{sample.image.name}"
    if name.lower() in occupied:
        name = f"{sample.digest[:12]}_{sample.image.name}"
    occupied.add(name.lower())
    return name


def prepare_dataset(sources: list[Path], output: Path, seed: int = 42) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    samples, duplicates = collect_samples(sources)
    splits = grouped_split(samples, seed)
    manifest: list[dict[str, object]] = []
    occupied: set[str] = set()
    output.mkdir(parents=True)
    try:
        for split, split_samples in splits.items():
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for sample in split_samples:
                filename = _unique_name(sample, occupied)
                image_target = image_dir / filename
                try:
                    os.link(sample.image, image_target)
                except OSError:
                    shutil.copy2(sample.image, image_target)
                y_center = (1.0 + sample.top) / 2.0
                # Keep a tiny numerical margin inside the normalized image so
                # strict validators do not reject decimal round-trip noise.
                height = 2.0 * (1.0 - y_center) - 1e-9
                (label_dir / f"{Path(filename).stem}.txt").write_text(
                    f"0 0.5000000000 {y_center:.10f} 1.0000000000 {height:.10f}\n",
                    encoding="utf-8",
                )
                manifest.append(
                    {
                        "file": filename,
                        "source": str(sample.image),
                        "sha256": sample.digest,
                        "group": sample.group,
                        "split": split,
                        "top": sample.top,
                    }
                )
        (output / "dataset.yaml").write_text(
            "path: datasets/ultrasound_roi_v4\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n  0: ultrasound_roi\n",
            encoding="utf-8",
        )
        summary: dict[str, object] = {
            "images": len(samples),
            "duplicates_removed": len(duplicates),
            "groups": len({sample.group for sample in samples}),
            "splits": {name: len(items) for name, items in splits.items()},
            "seed": seed,
            "policy": "preserve annotated top; force left=0, right=1, bottom=1",
            "duplicates": duplicates,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception:
        shutil.rmtree(output)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_dataset(
        [path.expanduser().resolve() for path in args.source],
        args.output.expanduser().resolve(),
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
