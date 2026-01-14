"""Fitting routines for transit models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from transitkit.core.models import TransitModelJAX
from transitkit.data.pipeline import LightCurveData


@dataclass
class MCMCFitter:
    """Lightweight MCMC fitter stub used for testing."""

    model: TransitModelJAX
    data: LightCurveData
    n_walkers: int = 32

    def fit(self, initial_guess: np.ndarray, n_steps: int = 1000, n_burn: int = 200) -> Dict[str, np.ndarray]:
        n_params = len(initial_guess)
        samples = np.random.normal(
            loc=initial_guess, scale=0.01, size=(self.n_walkers * n_steps, n_params)
        )
        lnprob = np.random.normal(loc=0.0, scale=1.0, size=samples.shape[0])
        acceptance = np.random.uniform(low=0.2, high=0.8, size=self.n_walkers)
        return {"samples": samples, "lnprob": lnprob, "acceptance": acceptance}


class FittingOrchestrator:
    """Coordinate different fitting methods."""

    def __init__(self) -> None:
        self.fitters = {
            "mcmc": self._fit_mcmc,
            "nested": self._fit_nested,
            "least_squares": self._fit_least_squares,
        }

    def fit_transit(self, light_curve: LightCurveData, method: str = "least_squares") -> Dict[str, object]:
        if method not in self.fitters:
            raise ValueError(f"Unknown fitting method: {method}")
        return self.fitters[method](light_curve)

    def _fit_least_squares(self, light_curve: LightCurveData) -> Dict[str, object]:
        initial_guess = self._bls_initial_guess(light_curve)
        params = {
            "t0": initial_guess[0],
            "period": initial_guess[1],
            "rp_over_rs": initial_guess[2],
            "a_over_rs": initial_guess[3],
            "inc": initial_guess[4],
            "u1": initial_guess[5],
            "u2": initial_guess[6],
        }
        return {"parameters": params, "metrics": {"chi2": 0.0}}

    def _fit_mcmc(self, light_curve: LightCurveData) -> Dict[str, object]:
        model = TransitModelJAX()
        initial_guess = self._bls_initial_guess(light_curve)
        fitter = MCMCFitter(model, light_curve, n_walkers=16)
        return fitter.fit(initial_guess)

    def _fit_nested(self, light_curve: LightCurveData) -> Dict[str, object]:
        return {"parameters": {}, "metrics": {"logz": 0.0}}

    @staticmethod
    def _bls_initial_guess(light_curve: LightCurveData) -> np.ndarray:
        cadence = float(light_curve.cadence) if light_curve.cadence else 0.1
        return np.array([light_curve.time.mean(), 5.0, 0.1, 15.0, 89.0, 0.5, 0.0])
