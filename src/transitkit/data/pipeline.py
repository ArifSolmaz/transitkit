"""Light curve data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LightCurveData:
    """Container for light curve data."""

    time: Iterable[float]
    flux: Iterable[float]
    flux_err: Iterable[float]
    quality: Iterable[float]
    cadence: float
    mission: str
    target_id: str

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.flux_err = np.asarray(self.flux_err, dtype=float)
        self.quality = np.asarray(self.quality, dtype=float)
