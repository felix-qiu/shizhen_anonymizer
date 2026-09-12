"""Read and validate the single-class ultrasound ROI YOLO dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.errors import CleanerError
from src.image_io import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    def is_valid(self) -> bool:
        if self.class_id != 0 or self.width <= 0 or self.height <= 0:
            return False
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2))

    def to_xyxy(self, image_width: int, image_height: int) -> list[float]:
        return [
            (self.x_center - self.width / 2) * image_width,
            (self.y_center - self.height / 2) * image_height,
            (self.x_center + self.width / 2) * image_width,
            (self.y_center + self.height / 2) * image_height,
        ]


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    split: str
    image_path: Path
    label_path: Path
    annotations: tuple[YoloAnnotation, ...]
    manufacturer: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    split: str
    image: str
    label: str
    error: str
    detail: str


def load_metadata(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None or not Path(path).is_file():
        return {}
    metadata_path = Path(path)
    text = metadata_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        parsed: Any = json.loads(text)
        entries = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]

    result: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("image"):
            raise ValueError("each metadata entry must contain an image field")
        values = {
            key: str(entry[key])
            for key in ("manufacturer", "model")
            if entry.get(key) is not None
        }
        image_key = Path(str(entry["image"])).as_posix()
        result[image_key] = values
        result.setdefault(Path(image_key).name, values)
    return result


def read_annotations(path: str | Path) -> tuple[YoloAnnotation, ...]:
    label_path = Path(path)
    annotations: list[YoloAnnotation] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid numeric value") from exc
        annotations.append(YoloAnnotation(class_id, *coordinates))
    return tuple(annotations)


def _metadata_for(
    image_path: Path, dataset_root: Path, metadata: dict[str, dict[str, str]]
) -> dict[str, str]:
    relative = image_path.relative_to(dataset_root).as_posix()
    return metadata.get(relative, metadata.get(image_path.name, {}))


def scan_dataset(
    dataset_root: str | Path,
    metadata_path: str | Path | None = None,
    *,
    strict: bool = True,
) -> list[DatasetRecord]:
    root = Path(dataset_root).resolve()
    metadata = load_metadata(metadata_path or root / "metadata.json")
    records: list[DatasetRecord] = []
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                annotations = (
                    read_annotations(label_path) if label_path.is_file() else ()
                )
            except (OSError, ValueError):
                if strict:
                    raise
                annotations = ()
            values = _metadata_for(image_path, root, metadata)
            records.append(
                DatasetRecord(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    annotations=annotations,
                    manufacturer=values.get("manufacturer"),
                    model=values.get("model"),
                )
            )
    return records


def validate_dataset(dataset_root: str | Path) -> list[DatasetValidationIssue]:
    root = Path(dataset_root).resolve()
    issues: list[DatasetValidationIssue] = []
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            issues.append(
                DatasetValidationIssue(
                    split,
                    "",
                    "",
                    "MISSING_DIRECTORY",
                    "images or labels directory missing",
                )
            )
            continue

        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if not images:
            issues.append(
                DatasetValidationIssue(
                    split,
                    "",
                    "",
                    "EMPTY_SPLIT",
                    "split contains no supported images",
                )
            )
        for stem, image_path in sorted(images.items()):
            label_path = labels.get(stem, label_dir / f"{stem}.txt")
            try:
                load_image(image_path)
            except CleanerError as exc:
                issues.append(
                    DatasetValidationIssue(
                        split,
                        image_path.name,
                        label_path.name,
                        "INVALID_IMAGE",
                        exc.code.value,
                    )
                )
            if stem not in labels:
                issues.append(
                    DatasetValidationIssue(
                        split,
                        image_path.name,
                        label_path.name,
                        "LABEL_NOT_FOUND",
                        "matching label file does not exist",
                    )
                )
                continue
            try:
                annotations = read_annotations(label_path)
            except (OSError, ValueError) as exc:
                issues.append(
                    DatasetValidationIssue(
                        split,
                        image_path.name,
                        label_path.name,
                        "INVALID_LABEL",
                        str(exc),
                    )
                )
                continue
            if not annotations:
                issues.append(
                    DatasetValidationIssue(
                        split,
                        image_path.name,
                        label_path.name,
                        "EMPTY_ANNOTATION",
                        "label file contains no ROI",
                    )
                )
            elif len(annotations) > 1:
                issues.append(
                    DatasetValidationIssue(
                        split,
                        image_path.name,
                        label_path.name,
                        "MULTIPLE_ROI",
                        "exactly one ultrasound ROI annotation is required",
                    )
                )
            for index, annotation in enumerate(annotations):
                if not annotation.is_valid():
                    issues.append(
                        DatasetValidationIssue(
                            split,
                            image_path.name,
                            label_path.name,
                            "BBOX_OUT_OF_RANGE",
                            f"annotation {index} is outside normalized image bounds",
                        )
                    )
        for stem, label_path in sorted(labels.items()):
            if stem not in images:
                issues.append(
                    DatasetValidationIssue(
                        split,
                        "",
                        label_path.name,
                        "IMAGE_NOT_FOUND",
                        "matching image file does not exist",
                    )
                )
    return issues
