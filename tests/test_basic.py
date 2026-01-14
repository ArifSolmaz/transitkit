"""Comprehensive tests for TransitKit v2.0"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Test basic imports
def test_version():
    """Test version is set"""
    from transitkit import __version__
    assert __version__ == "2.0.0"

def test_hello():
    """Test hello function"""
    from transitkit import hello
    result = hello()
    assert "TransitKit" in result
    assert "2.0.0" in result

def test_module_imports():
    """Test all module imports"""
    import transitkit as tk
    
    # Test core module access
    assert hasattr(tk, 'core')
    assert hasattr(tk, 'analysis')
    assert hasattr(tk, 'visualization')
    assert hasattr(tk, 'io')
    assert hasattr(tk, 'utils')
    assert hasattr(tk, 'validation')
    assert hasattr(tk, 'nea')
    
    # Test backward compatibility functions
    assert callable(tk.hello)
    assert callable(tk.generate_transit_signal)
    assert callable(tk.add_noise)
    assert callable(tk.find_transits_box)
    assert callable(tk.lookup_planet)
    
    # Test new convenience functions
    assert callable(tk.load_tess_data)
    assert callable(tk.generate_transit_signal_advanced)
    assert callable(tk.find_transits_multiple)
    assert callable(tk.detrend_gp)
    assert callable(tk.create_transit_report)
    assert callable(tk.estimate_parameters_mcmc)
    assert callable(tk.validate_transit_detection)
    assert callable(tk.measure_ttvs)

def test_core_module():
    """Test core module functionality"""
    from transitkit.core import (
        TransitParameters,
        generate_transit_signal_mandel_agol,
        find_transits_bls_advanced,
        find_transits_multiple_methods,
        add_noise
    )
    
    # Test TransitParameters dataclass
    params = TransitParameters(
        period=10.0,
        t0=5.0,
        duration=0.1,
        depth=0.01,
        snr=15.5,
        fap=0.001
    )
    
    assert params.period == 10.0
    assert params.t0 == 5.0
    assert params.duration == 0.1
    assert params.depth == 0.01
    assert params.snr == 15.5
    assert params.fap == 0.001
    
    # Test to_dict method
    params_dict = params.to_dict()
    assert 'period' in params_dict
    assert 't0' in params_dict
    assert 'duration' in params_dict
    assert 'depth' in params_dict
    
    # Test from_bls_result method
    mock_bls_result = type('MockBLS', (), {
        'period': 5.0,
        'transit_time': 100.0,
        'depth': 0.02,
        'duration': 0.15,
        'snr': 20.0
    })()
    
    params2 = TransitParameters.from_bls_result(mock_bls_result, None, None)
    assert params2.period == 5.0
    assert params2.t0 == 100.0
    assert params2.depth == 0.02
    assert params2.duration == 0.15
    assert params2.snr == 20.0

def test_signal_generation():
    """Test transit signal generation"""
    from transitkit.core import generate_transit_signal_mandel_agol, add_noise
    
    # Create time array
    time = np.linspace(0, 30, 1000)
    
    # Test basic generation
    flux = generate_transit_signal_mandel_agol(
        time=time,
        period=10.0,
        t0=5.0,
        rprs=0.1,  # sqrt(0.01)
        aRs=10.0,
        u1=0.1,
        u2=0.3
    )
    
    assert len(flux) == len(time)
    assert np.all(flux <= 1.0)  # Flux should be normalized
    assert np.all(flux >= 0.0)  # Should be positive
    
    # Test noise addition
    flux_noisy = add_noise(flux, noise_level=0.001)
    assert len(flux_noisy) == len(flux)
    
    # Check that noise was added (std should be close to noise_level)
    noise = flux_noisy - flux
    assert np.std(noise) > 0.0001  # Should have some noise

def test_bls_analysis():
    """Test BLS analysis functionality"""
    from transitkit.core import find_transits_bls_advanced
    
    # Create synthetic data with a known transit
    time = np.linspace(0, 30, 2000)
    flux = np.ones_like(time)
    
    # Add a transit at period 10.0 days
    period = 10.0
    t0 = 5.0
    depth = 0.01
    duration = 0.1
    
    for i in range(4):
        tc = t0 + i * period
        in_transit = (time > tc - duration/2) & (time < tc + duration/2)
        flux[in_transit] = 1 - depth
    
    # Add small noise
    flux += np.random.normal(0, 0.0005, len(flux))
    
    # Run BLS
    result = find_transits_bls_advanced(
        time, flux,
        min_period=5.0,
        max_period=15.0,
        n_periods=1000
    )
    
    # Check result structure
    required_keys = ['period', 't0', 'duration', 'depth', 'snr', 'fap']
    for key in required_keys:
        assert key in result
    
    # Check that period is detected close to true period
    assert abs(result['period'] - period) < 0.5  # Within 0.5 days
    
    # Check that FAP is calculated
    assert 0 <= result['fap'] <= 1

def test_utils_module():
    """Test utilities module"""
    from transitkit.utils import (
        calculate_snr,
        estimate_limb_darkening,
        check_data_quality,
        detect_outliers_modified_zscore
    )
    
    # Test SNR calculation
    time = np.linspace(0, 30, 1000)
    flux = np.ones_like(time)
    
    # Add a fake transit
    period = 10.0
    t0 = 5.0
    duration = 0.1
    in_transit = ((time - t0) % period) < duration
    flux[in_transit] = 0.99
    
    snr = calculate_snr(time, flux, period, t0, duration)
    assert isinstance(snr, float)
    
    # Test limb darkening estimation
    u1, u2 = estimate_limb_darkening(5800, 4.4, 0.0, method='quadratic')
    assert 0 <= u1 <= 1
    assert 0 <= u2 <= 1
    
    # Test data quality check
    quality = check_data_quality(time, flux)
    assert 'n_points' in quality
    assert 'time_span' in quality
    assert 'flux_mean' in quality
    assert 'flux_std' in quality
    
    # Test outlier detection
    data = np.random.normal(0, 1, 100)
    data[0] = 100  # Add an obvious outlier
    outliers = detect_outliers_modified_zscore(data, threshold=3.5)
    assert np.sum(outliers) >= 1  # Should detect at least the obvious outlier

def test_validation_module():
    """Test validation module"""
    from transitkit.validation import (
        validate_transit_parameters,
        calculate_detection_significance
    )
    from transitkit.core import TransitParameters
    
    # Test parameter validation
    time = np.linspace(0, 30, 1000)
    
    # Valid parameters
    params = TransitParameters(
        period=10.0,
        t0=15.0,
        duration=0.1,
        depth=0.01,
        snr=15.0
    )
    
    validation = validate_transit_parameters(params, time, np.ones_like(time))
    assert 'period_positive' in validation
    assert 'duration_lt_period' in validation
    assert 't0_in_range' in validation
    assert 'all_passed' in validation
    
    # Test significance calculation
    mock_result = {
        'period': 10.0,
        't0': 15.0,
        'time': time,
        'flux': np.ones_like(time)
    }
    
    significance = calculate_detection_significance(mock_result, n_shuffles=10)
    assert 'p_value' in significance
    assert 'significance_sigma' in significance
    assert 'n_shuffles' in significance
    assert 0 <= significance['p_value'] <= 1

def test_io_module():
    """Test IO module functionality"""
    from transitkit.io import export_transit_results
    
    # Test export functionality
    test_results = {
        'period': 10.0,
        't0': 100.0,
        'duration': 0.1,
        'depth': 0.01,
        'snr': 15.5,
        'data': np.array([1, 2, 3]),
        'nested': {
            'value': 42,
            'array': np.array([4, 5, 6])
        }
    }
    
    # Test JSON export
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        export_transit_results(test_results, json_file, format='json')
        assert os.path.exists(json_file)
        assert os.path.getsize(json_file) > 0
    finally:
        if os.path.exists(json_file):
            os.unlink(json_file)
    
    # Test CSV export (should work with flattened data)
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        csv_file = f.name
    
    try:
        # Create simple data for CSV
        simple_results = {'period': [10.0], 'depth': [0.01]}
        export_transit_results(simple_results, csv_file, format='csv')
        assert os.path.exists(csv_file)
        assert os.path.getsize(csv_file) > 0
    finally:
        if os.path.exists(csv_file):
            os.unlink(csv_file)

def test_nea_module():
    """Test NEA module (mocked)"""
    from transitkit.nea import lookup_planet
    
    # Test that function exists and has correct signature
    import inspect
    sig = inspect.signature(lookup_planet)
    params = list(sig.parameters.keys())
    
    assert 'query_text' in params
    assert 'default_only' in params
    assert 'limit' in params
    
    # Function should be callable
    assert callable(lookup_planet)

def test_visualization_imports():
    """Test visualization module imports"""
    from transitkit.visualization import (
        setup_publication_style,
        plot_transit_summary,
        create_transit_report_figure
    )
    
    # Just test that functions are importable
    assert callable(setup_publication_style)
    assert callable(plot_transit_summary)
    assert callable(create_transit_report_figure)

def test_analysis_imports():
    """Test analysis module imports"""
    from transitkit.analysis import (
        detrend_light_curve_gp,
        measure_transit_timing_variations,
        calculate_transit_duration_ratio
    )
    
    # Just test that functions are importable
    assert callable(detrend_light_curve_gp)
    assert callable(measure_transit_timing_variations)
    assert callable(calculate_transit_duration_ratio)

def test_backward_compatibility():
    """Test backward compatibility functions"""
    import transitkit as tk
    
    # Test old function names still work
    time = np.linspace(0, 30, 100)
    
    # generate_transit_signal should work (with deprecation warning)
    with pytest.warns(DeprecationWarning):
        flux = tk.generate_transit_signal(time, period=10.0, depth=0.01, duration=0.1)
        assert len(flux) == len(time)
    
    # find_transits_box should work (with deprecation warning)
    flux = np.ones_like(time)
    with pytest.warns(DeprecationWarning):
        result = tk.find_transits_box(time, flux, min_period=1.0, max_period=20.0)
        assert 'period' in result
    
    # plot_light_curve should work (with deprecation warning)
    with pytest.warns(DeprecationWarning):
        # This returns a matplotlib figure
        fig = tk.plot_light_curve(time, flux)
        assert fig is not None

def test_mcmc_estimation():
    """Test MCMC parameter estimation (basic)"""
    from transitkit.core import estimate_parameters_mcmc
    
    # Create synthetic data
    np.random.seed(42)
    time = np.linspace(0, 30, 500)
    flux = np.ones_like(time)
    
    # Add a transit
    period_true = 10.0
    t0_true = 5.0
    duration_true = 0.1
    depth_true = 0.01
    
    for i in range(4):
        tc = t0_true + i * period_true
        in_transit = (time > tc - duration_true/2) & (time < tc + duration_true/2)
        flux[in_transit] = 1 - depth_true
    
    # Add noise
    flux += np.random.normal(0, 0.001, len(flux))
    flux_err = np.ones_like(flux) * 0.001
    
    # Run quick MCMC (small for testing)
    samples, errors = estimate_parameters_mcmc(
        time, flux, flux_err,
        period_guess=9.5,
        t0_guess=4.5,
        duration_guess=0.15,
        depth_guess=0.015,
        n_walkers=8,
        n_steps=100,
        burnin=20
    )
    
    # Check outputs
    assert samples is not None
    assert errors is not None
    assert 'period_err' in errors
    assert 't0_err' in errors
    assert 'duration_err' in errors
    assert 'depth_err' in errors

def test_multiple_methods():
    """Test multiple detection methods"""
    from transitkit.core import find_transits_multiple_methods
    
    # Create synthetic data
    time = np.linspace(0, 30, 1000)
    flux = np.ones_like(time)
    
    # Add a transit
    period = 10.0
    t0 = 5.0
    depth = 0.01
    duration = 0.1
    
    for i in range(4):
        tc = t0 + i * period
        in_transit = (time > tc - duration/2) & (time < tc + duration/2)
        flux[in_transit] = 1 - depth
    
    flux += np.random.normal(0, 0.0005, len(flux))
    
    # Test with BLS only
    results = find_transits_multiple_methods(
        time, flux,
        min_period=5.0,
        max_period=15.0,
        methods=['bls']
    )
    
    assert 'bls' in results
    assert 'consensus' in results
    assert 'validation' in results
    
    # Check consensus has period
    consensus = results['consensus']
    if consensus:
        assert 'period' in consensus

def test_gaussian_process():
    """Test Gaussian Process detrending"""
    from transitkit.analysis import detrend_light_curve_gp
    
    # Create synthetic data with trend
    time = np.linspace(0, 10, 200)
    trend = 0.01 * time  # Linear trend
    flux = 1.0 + trend + np.random.normal(0, 0.001, len(time))
    
    # Apply GP detrending
    flux_detrended, trend_fit, gp = detrend_light_curve_gp(time, flux)
    
    assert len(flux_detrended) == len(flux)
    assert len(trend_fit) == len(flux)
    assert gp is not None
    
    # Detrended flux should have less trend
    slope_original = np.polyfit(time, flux, 1)[0]
    slope_detrended = np.polyfit(time, flux_detrended, 1)[0]
    
    assert abs(slope_detrended) < abs(slope_original)

def test_transit_timing_variations():
    """Test TTV measurement"""
    from transitkit.analysis import measure_transit_timing_variations
    
    # Create data with TTVs
    time = np.linspace(0, 100, 2000)
    flux = np.ones_like(time)
    
    period = 10.0
    t0 = 5.0
    duration = 0.1
    depth = 0.01
    
    # Add transits with TTVs
    for i in range(10):
        # Add sinusoidal TTV
        ttv = 0.01 * np.sin(2 * np.pi * i / 5)  # ~1% period TTV
        tc = t0 + i * period + ttv
        
        in_transit = (time > tc - duration/2) & (time < tc + duration/2)
        flux[in_transit] = 1 - depth
    
    flux += np.random.normal(0, 0.0005, len(flux))
    
    # Measure TTVs
    ttv_results = measure_transit_timing_variations(
        time, flux, period, t0, duration
    )
    
    assert 'ttvs_detected' in ttv_results
    assert 'p_value' in ttv_results
    assert 'measurements' in ttv_results
    
    if ttv_results['measurements']:
        assert 'epoch' in ttv_results['measurements'][0]
        assert 'ttv' in ttv_results['measurements'][0]

def test_data_quality():
    """Test comprehensive data quality checks"""
    from transitkit.utils import check_data_quality
    
    # Create good quality data
    time = np.linspace(0, 30, 1000)
    flux = np.random.normal(1.0, 0.001, len(time))
    
    quality = check_data_quality(time, flux)
    
    assert quality['n_nans'] == 0
    assert quality['has_nans'] == False
    assert quality['is_sorted'] == True
    assert quality['n_points'] == 1000
    assert quality['time_span'] == 30.0
    
    # Create data with issues
    time_bad = np.array([0, 2, 1, 3, 4])  # Not sorted
    flux_bad = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
    
    quality_bad = check_data_quality(time_bad, flux_bad)
    
    assert quality_bad['has_nans'] == True
    assert quality_bad['is_sorted'] == False

@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="Skipping slow tests in CI")
def test_slow_mcmc():
    """Test full MCMC (marked as slow)"""
    from transitkit.core import estimate_parameters_mcmc
    
    # Smaller test for slow test
    time = np.linspace(0, 30, 200)
    flux = np.ones_like(time)
    flux_err = np.ones_like(flux) * 0.001
    
    samples, errors = estimate_parameters_mcmc(
        time, flux, flux_err,
        period_guess=10.0,
        t0_guess=5.0,
        duration_guess=0.1,
        depth_guess=0.01,
        n_walkers=16,
        n_steps=200,
        burnin=50
    )
    
    assert samples is not None
    assert errors is not None

def test_cli_import():
    """Test CLI module import"""
    from transitkit import cli
    
    # Test that CLI functions exist
    assert hasattr(cli, 'cli')
    assert hasattr(cli, 'main')
    
    # Test CLI group creation
    import click
    assert isinstance(cli.cli, click.Group)

def test_gui_import():
    """Test GUI module import"""
    from transitkit import gui_app
    
    # Test that GUI class exists
    assert hasattr(gui_app, 'TransitKitGUI')
    assert hasattr(gui_app, 'main')
    
    # Test GUI class can be instantiated
    # (We don't actually run it in tests)
    assert callable(gui_app.TransitKitGUI)

def test_deprecation_warnings():
    """Test that deprecated functions warn properly"""
    import warnings
    import transitkit as tk
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call deprecated function
        result = tk.generate_transit_signal([0, 1, 2], period=1.0, depth=0.01, duration=0.1)
        
        # Check that warning was issued
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

def test_package_structure():
    """Test overall package structure"""
    import transitkit as tk
    
    # Check all expected attributes exist
    expected_attrs = [
        '__version__', '__author__', '__email__', '__license__', '__citation__',
        'hello', 'generate_transit_signal', 'add_noise', 'plot_light_curve',
        'find_transits_box', 'lookup_planet', 'core', 'analysis', 'visualization',
        'io', 'utils', 'validation', 'nea', 'load_tess_data',
        'generate_transit_signal_advanced', 'find_transits_multiple', 'detrend_gp',
        'create_transit_report', 'estimate_parameters_mcmc', 'validate_transit_detection',
        'measure_ttvs'
    ]
    
    for attr in expected_attrs:
        assert hasattr(tk, attr), f"Missing attribute: {attr}"

def test_error_handling():
    """Test error handling in core functions"""
    from transitkit.core import find_transits_bls_advanced
    
    # Test with invalid inputs
    time = np.array([1, 2, 3])
    flux = np.array([1, 2])  # Wrong length
    
    with pytest.raises(ValueError):
        find_transits_bls_advanced(time, flux)
    
    # Test with all NaN data
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    
    with pytest.raises(ValueError):
        find_transits_bls_advanced(time, flux)

def test_numpy_compatibility():
    """Test that functions work with numpy arrays"""
    from transitkit.core import add_noise
    import numpy as np
    
    # Test with different dtypes
    for dtype in [np.float32, np.float64]:
        flux = np.ones(100, dtype=dtype)
        noisy = add_noise(flux, noise_level=0.001)
        
        assert noisy.dtype == np.float64  # Should be promoted to float64
        assert len(noisy) == len(flux)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])