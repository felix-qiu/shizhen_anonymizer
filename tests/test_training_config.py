from pathlib import Path

import pytest

from training.config import (
    load_dataset_config,
    load_training_config,
    validate_model_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_config_defines_ultrasound_roi_layout() -> None:
    config = load_dataset_config("datasets/ultrasound_roi/dataset.yaml", PROJECT_ROOT)

    assert config.root == PROJECT_ROOT / "datasets/ultrasound_roi"
    assert config.train_images == config.root / "images/train"
    assert config.val_images == config.root / "images/val"
    assert config.names == {0: "ultrasound_roi"}


def test_training_config_loads_phase_3_defaults() -> None:
    config = load_training_config("config/train.yaml", PROJECT_ROOT)

    assert config.model == "yolo11s.pt"
    assert config.model_version == "v1"
    assert config.epochs == 100
    assert config.imgsz == 640
    assert config.batch == 16
    assert config.device == "auto"
    assert config.output_dir == PROJECT_ROOT / "runs"
    assert config.run_name == "train"
    assert config.export_dir == PROJECT_ROOT / "models/trained"
    assert config.dataset_yaml == (
        PROJECT_ROOT / "datasets/ultrasound_roi/dataset.yaml"
    )


def test_model_version_rejects_path_components() -> None:
    with pytest.raises(ValueError, match="model version"):
        validate_model_version("../../escape")
