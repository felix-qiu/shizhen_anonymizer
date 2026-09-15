from pathlib import Path
from typing import Any

from service.dicom_service import DicomService


class FakeResponse:
    def __init__(self, status_code: int, chunks: list[bytes]) -> None:
        self.status_code = status_code
        self._chunks = chunks
        self.closed = False

    def iter_content(self, chunk_size: int):
        assert chunk_size == 8192
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None
        self.uploaded = b""

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.request = {"url": url, **kwargs}
        self.uploaded = kwargs["files"]["video"][1].read()
        return self.response


def test_dicom_service_uses_source_http_contract(tmp_path: Path) -> None:
    video = tmp_path / "clean.mp4"
    output = tmp_path / "clean.dcm"
    video.write_bytes(b"video-content")
    response = FakeResponse(200, [b"dicom-", b"content"])
    session = FakeSession(response)
    service = DicomService(
        "http://dicom.test/convert", timeout_seconds=12, session=session
    )

    success = service.convert(video, output, patient_id="patient-001")

    assert success is True
    assert output.read_bytes() == b"dicom-content"
    assert session.uploaded == b"video-content"
    assert session.request is not None
    assert session.request["url"] == "http://dicom.test/convert"
    assert session.request["data"] == {"patientId": "patient-001"}
    assert session.request["stream"] is True
    assert session.request["timeout"] == 12
    assert session.request["files"]["video"][:1] == ("video.avi",)
    assert session.request["files"]["video"][2] == "video/x-msvideo"
    assert response.closed is True


def test_dicom_service_keeps_no_partial_file_on_http_failure(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clean.mp4"
    output = tmp_path / "clean.dcm"
    video.write_bytes(b"video-content")
    response = FakeResponse(500, [])
    service = DicomService(
        "http://dicom.test/convert", session=FakeSession(response)
    )

    assert service.convert(video, output) is False
    assert not output.exists()
    assert not output.with_suffix(".dcm.part").exists()
