"""HTTP client for converting a cleaned video to DICOM."""

import logging
from pathlib import Path

import requests


class DicomService:
    """Stream a cleaned video to the external DICOM conversion service."""

    def __init__(
        self,
        server_url: str | None,
        timeout_seconds: float = 300.0,
        logger: logging.Logger | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._server_url = server_url
        self._timeout_seconds = timeout_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._session = session or requests.Session()

    def convert(
        self,
        video_path: str | Path,
        output_path: str | Path,
        patient_id: str | None = None,
    ) -> bool:
        source = Path(video_path)
        destination = Path(output_path)
        temporary = destination.with_suffix(f"{destination.suffix}.part")
        if not self._server_url:
            self._logger.warning("dicom_conversion_skipped reason=server_url_missing")
            return False
        if not source.is_file():
            self._logger.error("dicom_conversion_failed video_not_found=%s", source)
            return False

        try:
            temporary.unlink(missing_ok=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as video_stream:
                response = self._session.post(
                    self._server_url,
                    files={
                        "video": (
                            "video.avi",
                            video_stream,
                            "video/x-msvideo",
                        )
                    },
                    data={"patientId": patient_id} if patient_id else {},
                    stream=True,
                    timeout=self._timeout_seconds,
                )
                try:
                    if response.status_code != 200:
                        temporary.unlink(missing_ok=True)
                        self._logger.error(
                            "dicom_conversion_failed status=%d video=%s",
                            response.status_code,
                            source,
                        )
                        return False
                    with temporary.open("wb") as output_stream:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                output_stream.write(chunk)
                finally:
                    response.close()
            temporary.replace(destination)
            self._logger.info(
                "dicom_conversion_complete video=%s output=%s", source, destination
            )
            return True
        except Exception as exc:  # noqa: BLE001 - conversion must not abort the batch
            temporary.unlink(missing_ok=True)
            self._logger.error(
                "dicom_conversion_failed video=%s error=%s", source, exc
            )
            return False

    def close(self) -> None:
        self._session.close()
