from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.errors import CleanerError, ErrorCode
from training.config import TrainingConfig, require_model_file
from training.export import export_model
from training.train import train_model
from training.validate import validate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeTrainModel:
    def __init__(self, reference: str, save_dir: Path) -> None:
        self.reference = reference
        self.save_dir = save_dir
        self.arguments: dict[str, object] = {}

    def train(self, **kwargs: object) -> SimpleNamespace:
        self.arguments = kwargs
        with Path(str(kwargs["data"])).open("r", encoding="utf-8") as stream:
            self.dataset_config = yaml.safe_load(stream)
        return SimpleNamespace(save_dir=self.save_dir)


def test_train_model_passes_config_to_yolo(tmp_path: Path) -> None:
    fake = FakeTrainModel("yolo11s.pt", tmp_path / "runs/train")
    config = TrainingConfig(
        model="yolo11s.pt",
        model_version="test-v1",
        dataset_yaml=PROJECT_ROOT / "training/dataset.yaml",
        dataset_root=PROJECT_ROOT / "datasets/ultrasound_roi",
        epochs=12,
        imgsz=512,
        batch=4,
        output_dir=tmp_path / "runs",
        run_name="train",
        export_dir=tmp_path / "artifacts",
    )

    best = train_model(config, yolo_factory=lambda reference: fake)

    assert best == tmp_path / "runs/train/weights/best.pt"
    assert fake.arguments["epochs"] == 12
    assert fake.arguments["imgsz"] == 512
    assert fake.arguments["batch"] == 4
    assert fake.arguments["save"] is True
    assert fake.dataset_config["path"] == str(config.dataset_root)


def test_validate_model_returns_detection_metrics(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint")
    results = SimpleNamespace(
        results_dict={
            "metrics/mAP50(B)": 0.91,
            "metrics/mAP50-95(B)": 0.72,
            "metrics/precision(B)": 0.88,
            "metrics/recall(B)": 0.86,
        }
    )
    fake_model = SimpleNamespace(val=lambda **kwargs: results)

    metrics = validate_model(
        model_path,
        PROJECT_ROOT / "training/dataset.yaml",
        imgsz=640,
        yolo_factory=lambda reference: fake_model,
    )

    assert metrics.map50 == 0.91
    assert metrics.map50_95 == 0.72
    assert metrics.precision == 0.88
    assert metrics.recall == 0.86


def test_export_model_supports_pt_and_onnx(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint")
    artifact_dir = tmp_path / "artifacts"

    packaged_pt = export_model(
        model_path,
        "pt",
        640,
        artifact_dir,
        artifact_name="yolo11s_ultrasound_roi_test-v1.pt",
    )

    exported_onnx = tmp_path / "best.onnx"

    class FakeExportModel:
        def export(self, **kwargs: object) -> str:
            assert kwargs == {"format": "onnx", "imgsz": 640}
            exported_onnx.write_bytes(b"onnx")
            return str(exported_onnx)

    packaged_onnx = export_model(
        model_path,
        "onnx",
        640,
        artifact_dir,
        yolo_factory=lambda reference: FakeExportModel(),
        artifact_name="yolo11s_ultrasound_roi_test-v1.onnx",
    )

    assert packaged_pt.read_bytes() == b"checkpoint"
    assert packaged_onnx.read_bytes() == b"onnx"
    assert packaged_pt.name == "yolo11s_ultrasound_roi_test-v1.pt"
    assert packaged_onnx.name == "yolo11s_ultrasound_roi_test-v1.onnx"


def test_require_model_file_has_stable_error_code(tmp_path: Path) -> None:
    with pytest.raises(CleanerError) as error:
        require_model_file(tmp_path / "missing.pt")

    assert error.value.code is ErrorCode.MODEL_NOT_FOUND
