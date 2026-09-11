from pathlib import Path

from src.config import load_config


def test_phase_2_video_config_is_loaded() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = load_config(project_root / "config/config.yaml", project_root)

    assert config.video.detect_interval == 30
    assert config.video.output_fps is None
