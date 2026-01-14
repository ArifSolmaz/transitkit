"""Tests for transit analysis functions"""

import numpy as np
import transitkit as tk

def test_generate_transit_signal():
    """Test transit signal generation"""
    time = np.linspace(0, 10, 1000)
    flux = tk.generate_transit_signal(time, period=5.0, depth=0.01, duration=0.1)
    
    assert len(flux) == len(time)
    assert np.all(flux >= 0.98)  # Should be near 1.0
    assert np.any(flux < 0.995)  # Should have some transit dips

def test_add_noise():
    """Test noise addition"""
    flux = np.ones(100)
    noisy = tk.add_noise(flux, noise_level=0.01)
    
    assert len(noisy) == len(flux)
    assert np.std(noisy - flux) > 0  # Should have noise

def test_find_transits():
    """Test transit finding (simplified)"""
    time = np.linspace(0, 30, 3000)
    flux = tk.generate_transit_signal(time, period=5.0, depth=0.02)
    flux = tk.add_noise(flux, noise_level=0.001)
    
    results = tk.find_transits_box(time, flux, min_period=1.0, max_period=10.0)
    
    assert "all_power" in results
    assert "period" in results
