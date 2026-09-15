from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import create_app
from service.directory_service import DirectoryService
from service.image_service import ImageService
from service.server_config import ServerSettings
from service.video_service import VideoServiceResult
from src.detector.roi_detector import ROIDetector, ROIResult
from src.errors import CleanerError, ErrorCode


class FixedDetector(ROIDetector):
    def __init__(self, result: ROIResult | None) -> None:
        self.result = result

    def detect(self, image: np.ndarray) -> ROIResult | None:
        return self.result


class FakeVideoService:
    def process(self, input_path: Path, output_path: Path) -> VideoServiceResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clean-video")
        return VideoServiceResult(
            frames=42,
            confidence=0.97,
            inference_time_ms=12.0,
            total_time_seconds=0.5,
            output_path=output_path,
        )


class FakeDicomService:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, str | None]] = []

    def convert(
        self,
        video_path: Path,
        output_path: Path,
        patient_id: str | None = None,
    ) -> bool:
        self.calls.append((video_path, output_path, patient_id))
        output_path.write_bytes(b"dicom")
        return True


def make_app(tmp_path: Path):
    settings = ServerSettings(
        host="127.0.0.1",
        port=8000,
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        model_path=tmp_path / "model.pt",
        log_file=tmp_path / "logs/app.log",
    )
    return create_app(settings, load_model_on_startup=False), settings


def png_bytes() -> bytes:
    image = np.full((20, 30, 3), 127, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health_endpoint(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")
        docs_response = client.get("/docs")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert docs_response.status_code == 200
    assert "path=/health" in settings.log_file.read_text(encoding="utf-8")


def test_frontend_and_assets_are_served(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)

    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert page.status_code == 200
    assert "数据脱敏" in page.text
    assert "数据脱敏工作台" in page.text
    assert "导入数据文件夹" in page.text
    assert "脱敏当前文件" in page.text
    assert "全部脱敏" in page.text
    assert "文件列表" in page.text
    assert "/api/v1/image/clean" in script.text
    assert "/api/v1/video/clean" in script.text
    assert "webkitdirectory" in page.text
    assert "processSelection" in script.text
    assert "processCurrent" in script.text
    assert "navigateSelection" in script.text
    assert "A 上一个 · D 下一个" in page.text
    assert "file-search" in page.text
    assert "showItem" in script.text
    assert "FILE_TYPES" in script.text
    assert "lg:h-0" in page.text
    assert "createDocumentFragment" in script.text


def test_application_lifecycle_loads_model_once(tmp_path: Path, monkeypatch) -> None:
    load_calls = 0
    close_calls = 0

    class FakeModelManager:
        def __init__(self, **kwargs) -> None:
            pass

        def load(self) -> ROIDetector:
            nonlocal load_calls
            load_calls += 1
            return FixedDetector(ROIResult(0.98, [1, 1, 10, 10]))

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    monkeypatch.setattr(api_main, "ModelManager", FakeModelManager)
    _, settings = make_app(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert load_calls == 1
    assert close_calls == 1


def test_image_clean_endpoint_uses_service_layer(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)
    app.state.image_service = ImageService(
        FixedDetector(ROIResult(0.98, [2, 3, 28, 18])), output_size=32
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/clean",
            files={"file": ("test.png", png_bytes(), "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["confidence"] == 0.98
    assert payload["bbox"] == [0, 3, 30, 20]
    assert (settings.output_dir / payload["output"]).is_file()

    with TestClient(app) as client:
        output_response = client.get(f"/api/v1/output/{payload['output']}")

    assert output_response.status_code == 200
    assert output_response.headers["content-type"] == "image/png"
    assert output_response.content


def test_image_check_endpoint_reports_top_anonymization_need(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)
    app.state.image_service = ImageService(
        FixedDetector(ROIResult(0.96, [4, 3, 26, 18])), output_size=None
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/check",
            files={"file": ("test.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "needs_anonymization": True,
        "confidence": 0.96,
        "bbox": [0, 3, 30, 20],
        "top_crop_pixels": 3,
        "top_crop_ratio": 0.15,
    }
    assert list(settings.input_dir.iterdir()) == []
    assert list(settings.output_dir.iterdir()) == []


def test_image_check_endpoint_reports_no_top_crop(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(
        FixedDetector(ROIResult(0.99, [0, 0, 30, 20])), output_size=None
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/check",
            files={"file": ("test.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["needs_anonymization"] is False
    assert response.json()["top_crop_pixels"] == 0
    assert response.json()["top_crop_ratio"] == 0.0


def test_image_check_endpoint_does_not_treat_missing_roi_as_safe(
    tmp_path: Path,
) -> None:
    app, settings = make_app(tmp_path)
    app.state.image_service = ImageService(FixedDetector(None), output_size=None)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/check",
            files={"file": ("test.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 422
    assert response.json() == {"success": False, "error": "ROI_NOT_FOUND"}
    assert list(settings.input_dir.iterdir()) == []


def test_output_endpoint_rejects_unknown_file(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/v1/output/missing.png")
        invalid = client.get("/api/v1/output/not-allowed.txt")

    assert response.status_code == 404
    assert response.json() == {"success": False, "error": "FILE_NOT_FOUND"}
    assert invalid.status_code == 404


def test_video_clean_endpoint_uses_service_layer(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.video_service = FakeVideoService()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video/clean",
            files={"file": ("test.mp4", b"fake-video", "video/mp4")},
        )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["frames"] == 42
    assert response.json()["output"].endswith("_clean.mp4")


def test_crop_directory_processes_supported_files_recursively(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(
        FixedDetector(ROIResult(0.98, [2, 3, 28, 18])), output_size=None
    )
    app.state.video_service = FakeVideoService()
    app.state.directory_service = DirectoryService(
        app.state.image_service, app.state.video_service, FakeDicomService()
    )
    source = tmp_path / "study"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "first.png").write_bytes(png_bytes())
    (nested / "second.jpg").write_bytes(png_bytes())
    (source / "notes.txt").write_text("ignored", encoding="utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/crop/directory",
            json={"path": str(source), "file_type": "directory"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload == {
        "status": "success",
        "message": "Directory processed successfully",
        "data": {"directory_path": str(source)},
    }
    output_root = tmp_path / "cropped" / "study"
    assert (output_root / "first.png").is_file()
    assert (output_root / "nested" / "second.jpg").is_file()
    assert not (output_root / "notes.txt").exists()


def test_crop_directory_continues_after_file_failure(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(
        FixedDetector(ROIResult(0.98, [2, 3, 28, 18])), output_size=None
    )
    app.state.video_service = FakeVideoService()
    app.state.directory_service = DirectoryService(
        app.state.image_service, app.state.video_service, FakeDicomService()
    )
    source = tmp_path / "study"
    source.mkdir()
    (source / "broken.png").write_bytes(b"not-an-image")
    (source / "valid.png").write_bytes(png_bytes())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/crop/directory", json={"path": str(source)}
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload == {
        "status": "success",
        "message": "Directory processed successfully",
        "data": {"directory_path": str(source)},
    }
    assert (tmp_path / "cropped" / "study" / "valid.png").is_file()


def test_crop_directory_converts_cleaned_video_to_dicom(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    image_service = ImageService(FixedDetector(None), output_size=None)
    video_service = FakeVideoService()
    dicom_service = FakeDicomService()
    app.state.directory_service = DirectoryService(
        image_service, video_service, dicom_service
    )
    source = tmp_path / "study"
    source.mkdir()
    (source / "patient-001.mp4").write_bytes(b"source-video")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/crop/directory", json={"path": str(source)}
        )

    output_root = tmp_path / "cropped" / "study"
    assert response.status_code == 200
    assert (output_root / "patient-001.mp4").read_bytes() == b"clean-video"
    assert (output_root / "patient-001.dcm").read_bytes() == b"dicom"
    assert dicom_service.calls == [
        (
            output_root / "patient-001.mp4",
            output_root / "patient-001.dcm",
            "patient-001",
        )
    ]


def test_crop_directory_validates_server_path(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(FixedDetector(None), output_size=None)
    app.state.video_service = FakeVideoService()
    app.state.directory_service = DirectoryService(
        app.state.image_service, app.state.video_service, FakeDicomService()
    )
    file_path = tmp_path / "image.png"
    file_path.write_bytes(png_bytes())

    with TestClient(app) as client:
        empty = client.post("/api/v1/crop/directory", json={"path": ""})
        missing = client.post(
            "/api/v1/crop/directory", json={"path": str(tmp_path / "missing")}
        )
        not_directory = client.post(
            "/api/v1/crop/directory", json={"path": str(file_path)}
        )

    assert empty.status_code == 400
    assert empty.json() == {"detail": "Path is required"}
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Directory not found"}
    assert not_directory.status_code == 400
    assert not_directory.json() == {"detail": "Path must be a directory"}


def test_invalid_extension_has_unified_error(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(FixedDetector(None), output_size=32)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/clean",
            files={"file": ("test.txt", b"invalid", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "INVALID_FILE"}


def test_invalid_image_content_has_unified_error(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(FixedDetector(None), output_size=32)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/clean",
            files={"file": ("broken.png", b"not-an-image", "image/png")},
        )

    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "INVALID_FILE"}


def test_missing_upload_has_unified_error(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(FixedDetector(None), output_size=32)

    with TestClient(app) as client:
        response = client.post("/api/v1/image/clean")

    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "INVALID_FILE"}


def test_roi_not_found_has_unified_error(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    app.state.image_service = ImageService(FixedDetector(None), output_size=32)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/clean",
            files={"file": ("test.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 422
    assert response.json() == {"success": False, "error": "ROI_NOT_FOUND"}


def test_legacy_processing_error_is_mapped_to_api_error(tmp_path: Path) -> None:
    class BrokenImageService:
        def process(self, input_path: Path, output_path: Path) -> None:
            raise CleanerError(ErrorCode.IMAGE_LOAD_FAILED)

    app, _ = make_app(tmp_path)
    app.state.image_service = BrokenImageService()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/image/clean",
            files={"file": ("test.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "INVALID_FILE"}
