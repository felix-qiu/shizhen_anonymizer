from pathlib import Path

from service.server_config import load_server_settings


def test_server_config_resolves_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[1]

    settings = load_server_settings("config/server.yaml", project_root)

    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.input_dir == project_root / "storage/input"
    assert settings.output_dir == project_root / "storage/output"
    assert settings.model_path == project_root / "models/yolo11s_roi.pt"
    assert settings.log_file == project_root / "logs/app.log"
    assert settings.dicom_server_url == "http://192.168.55.7:8180/dicom/convert"
    assert settings.dicom_timeout_seconds == 300
