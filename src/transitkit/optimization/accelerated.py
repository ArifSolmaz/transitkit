"""Optional GPU-accelerated transit model."""

from __future__ import annotations

from typing import Dict, Iterable

import importlib.util

import numpy as np

from transitkit.core.models import TransitModelJAX


class AcceleratedTransitModel:
    """Wrapper that optionally enables GPU acceleration."""

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        if use_gpu:
            if importlib.util.find_spec("jax") is None:
                raise ImportError("GPU acceleration not available")
        self._model = TransitModelJAX()

    def compute(self, time: Iterable[float], params: Dict[str, float]) -> np.ndarray:
        return self._model.compute(time, params)
