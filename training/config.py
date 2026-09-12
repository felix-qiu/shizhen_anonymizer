"""Configuration and path validation shared by Phase 3 commands."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from src.errors import CleanerError, ErrorCode


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    yaml_path: Path
    root: Path
    train_images: Path
    val_images: Path
    names: dict[int, str]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model: str
    model_version: str
    dataset_yaml: Path
    dataset_root: Path
    epochs: int
    imgsz: int
    batch: int
    device: str
    output_dir: Path
    run_name: str
    export_dir: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream) or {}
    if not isinstance(content, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return content


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def validate_model_version(value: str) -> str:
    version = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError(
            "model version must contain only letters, numbers, dots, underscores, or hyphens"
        )
    return version


def load_dataset_config(
    yaml_path: str | Path, project_root: str | Path
) -> DatasetConfig:
    project = Path(project_root).resolve()
    path = _project_path(yaml_path, project).resolve()
    raw = _load_yaml(path)

    root = _project_path(raw["path"], project).resolve()
    names_raw = raw.get("names", {})
    if not isinstance(names_raw, dict):
        raise TypeError("dataset names must be a class-id mapping")
    names = {int(class_id): str(name) for class_id, name in names_raw.items()}
    if names != {0: "ultrasound_roi"}:
        raise ValueError("dataset must define only class 0: ultrasound_roi")

    return DatasetConfig(
        yaml_path=path,
        root=root,
        train_images=root / raw["train"],
        val_images=root / raw["val"],
        names=names,
    )


def load_training_config(
    config_path: str | Path, project_root: str | Path
) -> TrainingConfig:
    project = Path(project_root).resolve()
    raw = _load_yaml(_project_path(config_path, project).resolve())

    epochs = int(raw["train"]["epochs"])
    imgsz = int(raw["train"]["imgsz"])
    batch = int(raw["train"]["batch"])
    if min(epochs, imgsz, batch) <= 0:
        raise ValueError("epochs, imgsz, and batch must be positive integers")

    return TrainingConfig(
        model=str(raw["model"]["name"]),
        model_version=validate_model_version(str(raw["model"]["version"])),
        dataset_yaml=_project_path(raw["dataset"]["yaml"], project).resolve(),
        dataset_root=_project_path(raw["dataset"]["path"], project).resolve(),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=str(raw["train"].get("device", "auto")),
        output_dir=_project_path(raw["output"]["dir"], project).resolve(),
        run_name=str(raw["output"].get("name", "train")),
        export_dir=_project_path(raw["export"]["dir"], project).resolve(),
    )


def require_model_file(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise CleanerError(ErrorCode.MODEL_NOT_FOUND, str(path))
    return path


@contextmanager
def resolved_dataset_yaml(dataset: DatasetConfig) -> Iterator[Path]:
    """Create a temporary Ultralytics YAML with an absolute dataset root."""

    content = _load_yaml(dataset.yaml_path)
    content["path"] = str(dataset.root)
    with TemporaryDirectory(prefix="shizhen-dataset-") as directory:
        path = Path(directory) / "dataset.yaml"
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(content, stream, allow_unicode=True, sort_keys=False)
        yield path
