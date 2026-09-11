"""Validated YAML configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class VideoConfig:
    detect_interval: int
    output_fps: float | None


@dataclass(frozen=True, slots=True)
class AppConfig:
    model_path: Path
    device: str
    confidence_threshold: float
    output_size: int | tuple[int, int] | None
    video: VideoConfig


def _parse_output_size(value: Any) -> int | tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    ):
        return value[0], value[1]
    raise ValueError("output.size must be a positive integer, [width, height], or null")


def _parse_video_config(value: Any) -> VideoConfig:
    raw = value or {}
    detect_interval = int(raw.get("detect_interval", 30))
    if detect_interval <= 0:
        raise ValueError("video.detect_interval must be a positive integer")

    output_fps_value = raw.get("output_fps", "same")
    if isinstance(output_fps_value, str) and output_fps_value.lower() == "same":
        output_fps = None
    else:
        output_fps = float(output_fps_value)
        if output_fps <= 0:
            raise ValueError("video.output_fps must be 'same' or a positive number")
    return VideoConfig(detect_interval=detect_interval, output_fps=output_fps)


def load_config(config_path: str | Path, project_root: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    model_path = Path(raw["model"]["path"])
    if not model_path.is_absolute():
        model_path = Path(project_root) / model_path
    threshold = float(raw["confidence"]["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence.threshold must be between 0 and 1")

    return AppConfig(
        model_path=model_path,
        device=str(raw["device"]["type"]),
        confidence_threshold=threshold,
        output_size=_parse_output_size(raw["output"].get("size")),
        video=_parse_video_config(raw.get("video")),
    )
