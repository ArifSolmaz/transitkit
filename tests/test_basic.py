"""
TransitKit v2.0 - Comprehensive Test Suite

Tests for all core functionality including transit detection,
parameter estimation, analysis, and validation.
"""

import numpy as np
import pytest
import warnings

warnings.filterwarnings("ignore")

# Import transitkit modules
from transitkit.core import (
    TransitParameters,
    generate_transit_signal_mandel_agol,
    add_noise,
    find_transits_bls_advanced,
    find_transits_multiple_methods,
    find_period_gls,
    find_period_pdm,
    _phase_dispersion_theta,
)
from transitkit.analysis import (
    detrend_light_curve_gp,
    remove_systematics_pca,
    measure_transit_timing_variations,
)
from transitkit.utils import (
    calculate_snr,
    estimate_limb_darkening,
    calculate_transit_duration_from_parameters,
    check_data_quality,
    time_to_phase,
    detect_outliers_modified_zscore,
)
from transitkit.validation import (
    validate_transit_parameters,
    perform_injection_recovery_test,
    calculate_detection_significance,
)
from transitkit.io import export_transit_results


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_transit():
    """Generate synthetic transit light curve for testing."""
    np.random.seed(42)
    time = np.linspace(0, 50, 3000)
    period = 5.0
    t0 = 2.5
    depth = 0.01
    
    flux = generate_transit_signal_mandel_agol(
        time, period=period, t0=t0, depth=depth
    )
    flux_noisy = add_noise(flux, noise_level=0.001)
    
    return {
        "time": time,
        "flux": flux,
        "flux_noisy": flux_noisy,
        "period": period,
        "t0": t0,
        "depth": depth,
    }


@pytest.fixture
def transit_params():
    """Create TransitParameters instance for testing."""
    return TransitParameters(
        period=5.0,
        t0=2.5,
        duration=0.15,
        depth=0.01,
        period_err=0.001,
        t0_err=0.01,
        duration_err=0.01,
        depth_err=0.001,
        snr=50.0
    )


# =============================================================================
# Core Module Tests
# =============================================================================

class TestTransitParameters:
    """Tests for TransitParameters dataclass."""
    
    def test_creation(self):
        """Test basic parameter creation."""
        params = TransitParameters(
            period=5.0, t0=2.5, duration=0.15, depth=0.01
        )
        assert params.period == 5.0
        assert params.t0 == 2.5
        assert params.duration == 0.15
        assert params.depth == 0.01
    
    def test_rprs_derivation(self):
        """Test that rprs is derived from depth."""
        params = TransitParameters(
            period=5.0, t0=2.5, duration=0.15, depth=0.01
        )
        assert params.rprs == pytest.approx(0.1, rel=1e-6)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = TransitParameters(
            period=5.0, t0=2.5, duration=0.15, depth=0.01
        )
        d = params.to_dict()
        assert isinstance(d, dict)
        assert d["period"] == 5.0
        assert d["depth"] == 0.01


class TestSignalGeneration:
    """Tests for transit signal generation."""
    
    def test_generate_signal_basic(self):
        """Test basic signal generation."""
        time = np.linspace(0, 10, 500)
        flux = generate_transit_signal_mandel_agol(
            time, period=5.0, depth=0.01
        )
        
        assert len(flux) == len(time)
        assert np.all(flux <= 1.0)
        assert np.all(flux >= 0.99)  # Depth is 1%
    
    def test_generate_signal_with_rprs(self):
        """Test signal generation with rprs parameter."""
        time = np.linspace(0, 10, 500)
        flux = generate_transit_signal_mandel_agol(
            time, period=5.0, rprs=0.1
        )
        
        assert len(flux) == len(time)
        min_flux = np.min(flux)
        # rprs=0.1 means depth ~ 0.01
        assert min_flux < 1.0
    
    def test_generate_signal_empty_time(self):
        """Test signal generation with empty time array."""
        time = np.array([])
        flux = generate_transit_signal_mandel_agol(
            time, period=5.0, depth=0.01
        )
        assert len(flux) == 0
    
    def test_add_noise(self):
        """Test noise addition."""
        flux = np.ones(1000)
        noisy = add_noise(flux, noise_level=0.01, seed=42)
        
        assert len(noisy) == len(flux)
        assert np.std(noisy) == pytest.approx(0.01, rel=0.1)
    
    def test_add_noise_zero_level(self):
        """Test that zero noise level returns original."""
        flux = np.ones(100)
        noisy = add_noise(flux, noise_level=0.0)
        np.testing.assert_array_equal(flux, noisy)


class TestBLSDetection:
    """Tests for BLS transit detection."""
    
    def test_bls_detection(self, synthetic_transit):
        """Test BLS period detection accuracy."""
        result = find_transits_bls_advanced(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"],
            min_period=1.0,
            max_period=20.0
        )
        
        # Period should be within 1% of true value
        period_error = abs(result["period"] - synthetic_transit["period"])
        assert period_error / synthetic_transit["period"] < 0.01
    
    def test_bls_returns_required_keys(self, synthetic_transit):
        """Test that BLS returns all required keys."""
        result = find_transits_bls_advanced(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"]
        )
        
        required_keys = ["period", "t0", "duration", "depth", "snr", "fap"]
        for key in required_keys:
            assert key in result
    
    def test_bls_duration_period_constraint(self):
        """Test that BLS handles duration < period constraint."""
        time = np.linspace(0, 10, 500)
        flux = np.ones_like(time)
        
        # This should not raise an error even with small min_period
        result = find_transits_bls_advanced(
            time, flux,
            min_period=0.5,
            max_period=5.0
        )
        assert "period" in result


class TestPDM:
    """Tests for Phase Dispersion Minimization."""
    
    def test_pdm_theta_calculation(self):
        """Test PDM theta statistic calculation."""
        time = np.linspace(0, 100, 1000)
        # Sinusoidal signal with known period
        flux = 1.0 + 0.1 * np.sin(2 * np.pi * time / 10.0)
        
        # Theta should be low at true period
        theta_true = _phase_dispersion_theta(time, flux, period=10.0, nbins=10)
        theta_wrong = _phase_dispersion_theta(time, flux, period=7.3, nbins=10)
        
        assert theta_true < theta_wrong
    
    def test_pdm_period_detection(self, synthetic_transit):
        """Test PDM period detection."""
        result = find_period_pdm(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"],
            min_period=1.0,
            max_period=20.0
        )
        
        assert "period" in result
        assert "theta" in result
        assert result["method"] == "pdm"


class TestGLS:
    """Tests for Generalized Lomb-Scargle."""
    
    def test_gls_returns_required_keys(self, synthetic_transit):
        """Test GLS returns required keys."""
        result = find_period_gls(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"]
        )
        
        assert "period" in result
        assert "power" in result
        assert "fap" in result
        assert result["method"] == "gls"


class TestMultiMethod:
    """Tests for multi-method detection."""
    
    def test_multi_method_consensus(self, synthetic_transit):
        """Test multi-method consensus calculation."""
        result = find_transits_multiple_methods(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"],
            min_period=1.0,
            max_period=20.0,
            methods=["bls", "gls", "pdm"]
        )
        
        assert "bls" in result
        assert "gls" in result
        assert "pdm" in result
        assert "consensus" in result
        assert "period" in result["consensus"]


# =============================================================================
# Analysis Module Tests
# =============================================================================

class TestDetrending:
    """Tests for detrending functions."""
    
    def test_gp_detrending(self, synthetic_transit):
        """Test GP detrending."""
        detrended, trend, gp = detrend_light_curve_gp(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"]
        )
        
        assert len(detrended) == len(synthetic_transit["time"])
        assert len(trend) == len(synthetic_transit["time"])
    
    def test_pca_systematics(self, synthetic_transit):
        """Test PCA systematics removal."""
        result = remove_systematics_pca(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"],
            n_components=3
        )
        
        assert "corrected_flux" in result
        assert "explained_variance" in result
        assert len(result["corrected_flux"]) == len(synthetic_transit["time"])


class TestTTV:
    """Tests for TTV measurement."""
    
    def test_ttv_measurement(self, synthetic_transit):
        """Test TTV measurement returns required keys."""
        result = measure_transit_timing_variations(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"],
            period=synthetic_transit["period"],
            t0=synthetic_transit["t0"],
            duration=0.15
        )
        
        assert "ttvs_detected" in result
        assert "p_value" in result
        assert "ttvs" in result
        assert "epochs" in result
        assert "rms_ttv" in result
    
    def test_ttv_no_signal(self):
        """Test TTV measurement with flat light curve."""
        time = np.linspace(0, 100, 1000)
        flux = np.ones_like(time)
        
        result = measure_transit_timing_variations(
            time, flux, period=5.0, t0=2.5, duration=0.15
        )
        
        # Should not detect TTVs in flat data
        assert result["ttvs_detected"] == False


# =============================================================================
# Utils Module Tests
# =============================================================================

class TestUtils:
    """Tests for utility functions."""
    
    def test_calculate_snr(self, synthetic_transit):
        """Test SNR calculation."""
        snr = calculate_snr(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"],
            period=synthetic_transit["period"],
            t0=synthetic_transit["t0"],
            duration=0.15
        )
        
        assert snr > 0
        assert np.isfinite(snr)
    
    def test_estimate_limb_darkening(self):
        """Test limb darkening coefficient estimation."""
        # Solar-like star
        u1, u2 = estimate_limb_darkening(5800, 4.4, 0.0)
        assert 0 < u1 < 1
        assert 0 < u2 < 1
        
        # M-dwarf
        u1_m, u2_m = estimate_limb_darkening(3500, 4.8, 0.0)
        assert u1_m > u1  # M-dwarfs have stronger limb darkening
    
    def test_transit_duration_calculation(self):
        """Test transit duration calculation."""
        duration = calculate_transit_duration_from_parameters(
            period=5.0, aRs=10.0, rprs=0.1, b=0.0
        )
        
        assert duration > 0
        assert duration < 5.0  # Duration must be less than period
    
    def test_check_data_quality(self, synthetic_transit):
        """Test data quality check."""
        quality = check_data_quality(
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"]
        )
        
        assert "has_nans" in quality
        assert "n_points" in quality
        assert "is_sorted" in quality
        assert quality["n_points"] == len(synthetic_transit["time"])
    
    def test_time_to_phase(self):
        """Test time to phase conversion."""
        time = np.array([0, 2.5, 5.0, 7.5, 10.0])
        phase = time_to_phase(time, period=5.0, t0=0.0)
        
        expected = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(phase, expected)
    
    def test_outlier_detection(self):
        """Test modified Z-score outlier detection."""
        data = np.ones(100)
        data[50] = 10.0  # Obvious outlier
        
        outliers = detect_outliers_modified_zscore(data, threshold=3.5)
        
        assert outliers[50] == True
        assert np.sum(outliers) >= 1


# =============================================================================
# Validation Module Tests
# =============================================================================

class TestValidation:
    """Tests for validation functions."""
    
    def test_validate_transit_parameters(self, synthetic_transit, transit_params):
        """Test parameter validation."""
        validation = validate_transit_parameters(
            transit_params,
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"]
        )
        
        assert "period_positive" in validation
        assert "depth_positive" in validation
        assert "all_passed" in validation
        
        # Valid parameters should pass
        assert validation["period_positive"] == True
        assert validation["depth_positive"] == True
    
    def test_validate_invalid_parameters(self, synthetic_transit):
        """Test validation catches invalid parameters."""
        invalid_params = TransitParameters(
            period=-1.0,  # Invalid
            t0=2.5,
            duration=0.15,
            depth=0.01
        )
        
        validation = validate_transit_parameters(
            invalid_params,
            synthetic_transit["time"],
            synthetic_transit["flux_noisy"]
        )
        
        assert validation["period_positive"] == False
    
    @pytest.mark.slow
    def test_injection_recovery(self, synthetic_transit, transit_params):
        """Test injection-recovery analysis."""
        result = perform_injection_recovery_test(
            synthetic_transit["time"],
            transit_params,
            n_trials=5,
            noise_level=0.001
        )
        
        assert "recovery_rate" in result
        assert "n_trials" in result
        assert "n_recovered" in result
        assert 0 <= result["recovery_rate"] <= 1


# =============================================================================
# I/O Module Tests
# =============================================================================

class TestIO:
    """Tests for I/O functions."""
    
    def test_export_json(self, tmp_path):
        """Test JSON export."""
        results = {
            "period": 5.0,
            "t0": 2.5,
            "depth": 0.01,
            "array": np.array([1, 2, 3])
        }
        
        filepath = tmp_path / "results.json"
        export_transit_results(results, str(filepath), format="json")
        
        assert filepath.exists()
    
    def test_export_csv(self, tmp_path):
        """Test CSV export."""
        results = {
            "period": 5.0,
            "t0": 2.5,
            "depth": 0.01
        }
        
        filepath = tmp_path / "results.csv"
        export_transit_results(results, str(filepath), format="csv")
        
        assert filepath.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_full_analysis_workflow(self, synthetic_transit):
        """Test complete analysis workflow."""
        # 1. Generate signal (already done in fixture)
        time = synthetic_transit["time"]
        flux = synthetic_transit["flux_noisy"]
        true_period = synthetic_transit["period"]
        
        # 2. Detect transit
        bls_result = find_transits_bls_advanced(
            time, flux,
            min_period=1.0,
            max_period=20.0
        )
        
        # 3. Verify detection accuracy
        detected_period = bls_result["period"]
        period_error = abs(detected_period - true_period) / true_period
        assert period_error < 0.01  # Within 1%
        
        # 4. Calculate SNR
        snr = calculate_snr(
            time, flux,
            period=detected_period,
            t0=bls_result["t0"],
            duration=bls_result["duration"]
        )
        assert snr > 10  # Should have good SNR
        
        # 5. Create parameters object
        params = TransitParameters(
            period=detected_period,
            t0=bls_result["t0"],
            duration=bls_result["duration"],
            depth=bls_result["depth"],
            snr=snr
        )
        
        # 6. Validate parameters
        validation = validate_transit_parameters(params, time, flux)
        assert validation["all_passed"]
        
        # 7. Measure TTVs
        ttv_result = measure_transit_timing_variations(
            time, flux,
            period=detected_period,
            t0=bls_result["t0"],
            duration=bls_result["duration"]
        )
        assert "ttvs" in ttv_result


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_bls_short_data(self):
        """Test BLS with very short time series."""
        time = np.linspace(0, 1, 50)
        flux = np.ones_like(time)
        
        # Should not crash
        result = find_transits_bls_advanced(
            time, flux,
            min_period=0.1,
            max_period=0.5
        )
        assert "period" in result
    
    def test_bls_with_nans(self):
        """Test BLS handles NaN values."""
        time = np.linspace(0, 30, 1000)
        flux = np.ones_like(time)
        flux[100:110] = np.nan
        
        # Remove NaNs before BLS
        mask = np.isfinite(flux)
        result = find_transits_bls_advanced(
            time[mask], flux[mask]
        )
        assert "period" in result
    
    def test_ttv_insufficient_transits(self):
        """Test TTV with insufficient transits."""
        time = np.linspace(0, 2, 100)  # Only ~0.4 transits for P=5
        flux = np.ones_like(time)
        
        result = measure_transit_timing_variations(
            time, flux,
            period=5.0,
            t0=1.0,
            duration=0.1
        )
        
        # Should handle gracefully - may find 0-1 transits but not detect TTVs
        assert result["ttvs_detected"] == False
        assert len(result["ttvs"]) <= 1  # Can't measure TTVs with <=1 transit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
