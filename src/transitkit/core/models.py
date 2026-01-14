"""Transit model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable

import numpy as np

from transitkit.core.exceptions import TransitKitError


class LimbDarkeningLaw(str, Enum):
    """Supported limb darkening laws."""

    LINEAR = "linear"
    QUADRATIC = "quadratic"
    NONLINEAR = "nonlinear"


@dataclass
class TransitModelJAX:
    """Lightweight transit model with a Gaussian transit approximation."""

    law: LimbDarkeningLaw = LimbDarkeningLaw.QUADRATIC

    def compute(self, time: Iterable[float], params: Dict[str, float]) -> np.ndarray:
        """Compute a simple transit light curve."""
        time = np.asarray(time, dtype=float)
        self._validate_params(params)

        t0 = float(params["t0"])
        period = float(params["period"])
        rp_over_rs = float(params["rp_over_rs"])
        duration = float(params.get("duration", 0.1 * period))

        phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
        depth = rp_over_rs**2 * 1.02
        sigma = max(duration / 2.0, np.finfo(float).eps)
        transit_profile = np.exp(-0.5 * (phase / sigma) ** 2)
        flux = 1.0 - depth * transit_profile

        return np.clip(flux, 0.0, 1.1)

    @staticmethod
    def _validate_params(params: Dict[str, float]) -> None:
        required = {"t0", "period", "rp_over_rs", "a_over_rs", "inc"}
        missing = required - params.keys()
        if missing:
            raise TransitKitError(f"Missing required parameters: {sorted(missing)}")
        if params["period"] <= 0:
            raise TransitKitError("Period must be positive.")
        if params["rp_over_rs"] <= 0:
            raise TransitKitError("Radius ratio must be positive.")
        if params["a_over_rs"] <= 0:
            raise TransitKitError("Scaled semi-major axis must be positive.")
        if params["inc"] <= 0:
            raise TransitKitError("Inclination must be positive.")
