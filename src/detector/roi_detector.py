"""Detector-agnostic ROI contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ROIResult:
    """A detected region of interest in ``[x1, y1, x2, y2]`` format."""

    confidence: float
    bbox: list[int]


class ROIDetector(ABC):
    """Abstract interface implemented by all future ROI detectors."""

    @abstractmethod
    def detect(self, image: Any) -> ROIResult | None:
        """Return the best ROI, or ``None`` when no valid ROI is found."""
        raise NotImplementedError
