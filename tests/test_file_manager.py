from io import BytesIO
from pathlib import Path

import pytest

from service.file_manager import FileManager
from src.errors import CleanerError, ErrorCode


def test_file_manager_generates_unique_safe_names(tmp_path: Path) -> None:
    manager = FileManager(tmp_path / "input", tmp_path / "output")

    first = manager.save_upload(BytesIO(b"one"), "../../patient.png", "image")
    second = manager.save_upload(BytesIO(b"two"), "patient.png", "image")

    assert first.parent == tmp_path / "input"
    assert second.parent == tmp_path / "input"
    assert first != second
    assert first.suffix == ".png"
    assert "patient" not in first.name


def test_empty_upload_is_invalid(tmp_path: Path) -> None:
    manager = FileManager(tmp_path / "input", tmp_path / "output")

    with pytest.raises(CleanerError) as error:
        manager.save_upload(BytesIO(), "empty.png", "image")

    assert error.value.code is ErrorCode.INVALID_FILE
