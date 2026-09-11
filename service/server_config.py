"""Validated server configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ServerSettings:
    host: str
    port: int
    input_dir: Path
    output_dir: Path
    model_path: Path
    log_file: Path


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_server_settings(
    config_path: str | Path, project_root: str | Path
) -> ServerSettings:
    project = Path(project_root).resolve()
    path = _project_path(config_path, project)
    with path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}

    port = int(raw["server"]["port"])
    if not 1 <= port <= 65535:
        raise ValueError("server.port must be between 1 and 65535")
    return ServerSettings(
        host=str(raw["server"]["host"]),
        port=port,
        input_dir=_project_path(raw["storage"]["input_dir"], project).resolve(),
        output_dir=_project_path(raw["storage"]["output_dir"], project).resolve(),
        model_path=_project_path(raw["model"]["path"], project).resolve(),
        log_file=_project_path(raw["logging"]["file"], project).resolve(),
    )
