"""Pytest configuration and fixtures."""
import numpy as np
import pytest

@pytest.fixture
def synthetic_transit():
    """Generate synthetic transit data."""
    from transitkit.core import generate_transit_signal_mandel_agol, add_noise
    
    np.random.seed(42)
    time = np.linspace(0, 50, 3000)
    flux = generate_transit_signal_mandel_agol(time, period=5.0, t0=2.5, depth=0.01)
    flux_noisy = add_noise(flux, noise_level=0.001)
    
    return {"time": time, "flux": flux, "flux_noisy": flux_noisy, 
            "period": 5.0, "t0": 2.5, "depth": 0.01}

@pytest.fixture
def transit_params():
    """Create TransitParameters instance."""
    from transitkit.core import TransitParameters
    return TransitParameters(period=5.0, t0=2.5, duration=0.15, depth=0.01, snr=50.0)
