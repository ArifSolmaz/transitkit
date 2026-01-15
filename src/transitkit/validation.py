# validation.py - Validation and testing
"""
Validation tools for transit analysis results.
"""

import numpy as np
from scipy import stats
import warnings

# Import core functions needed for validation tests
from .core import (
    generate_transit_signal_mandel_agol, 
    add_noise, 
    find_transits_bls_advanced
)

def validate_transit_parameters(params, time, flux):
    """
    Validate transit parameters against physical limits.
    """
    validation = {}
    
    # Check period
    validation['period_positive'] = params.period > 0
    validation['period_realistic'] = 0.1 < params.period < 1000
    
    # Check duration
    validation['duration_positive'] = params.duration > 0
    validation['duration_lt_period'] = params.duration < params.period
    
    # Typical transit durations: 1-15 hours
    validation['duration_realistic'] = (
        1/24 < params.duration < 15/24
    )
    
    # Check depth
    validation['depth_positive'] = params.depth > 0
    validation['depth_lt_one'] = params.depth < 1
    
    # Typical depths: up to 3% (hot Jupiters)
    validation['depth_realistic'] = params.depth < 0.03
    
    # Check t0 within data range
    validation['t0_in_range'] = (
        time.min() <= params.t0 <= time.max()
    )
    
    # Calculate additional checks
    validation['snr_valid'] = params.snr > 0 if hasattr(params, 'snr') else True
    validation['fap_valid'] = 0 <= params.fap <= 1 if hasattr(params, 'fap') else True
    
    # Overall validation
    validation['all_passed'] = all(
        v for k, v in validation.items() 
        if not k.startswith('_') and isinstance(v, bool)
    )
    
    return validation


def compare_with_known_ephemeris(params, known_params, tolerance=0.01):
    """
    Compare detected parameters with known ephemeris.
    """
    comparison = {}
    
    # Period comparison
    if hasattr(known_params, 'period'):
        period_diff = abs(params.period - known_params.period) / known_params.period
        comparison['period_match'] = period_diff < tolerance
        comparison['period_diff'] = period_diff
    
    # T0 comparison (modulo period)
    if hasattr(known_params, 't0') and hasattr(known_params, 'period'):
        # Adjust t0 to same epoch
        epoch_diff = np.round((params.t0 - known_params.t0) / known_params.period)
        t0_aligned = known_params.t0 + epoch_diff * known_params.period
        t0_diff = abs(params.t0 - t0_aligned)
        comparison['t0_match'] = t0_diff < tolerance * known_params.period
        comparison['t0_diff'] = t0_diff
    
    # Depth comparison
    if hasattr(known_params, 'depth'):
        depth_diff = abs(params.depth - known_params.depth) / known_params.depth
        comparison['depth_match'] = depth_diff < tolerance
        comparison['depth_diff'] = depth_diff
    
    # Duration comparison
    if hasattr(known_params, 'duration'):
        dur_diff = abs(params.duration - known_params.duration) / known_params.duration
        comparison['duration_match'] = dur_diff < tolerance
        comparison['duration_diff'] = dur_diff
    
    # Overall match
    comparison['overall_match'] = all(
        v for k, v in comparison.items() 
        if k.endswith('_match') and isinstance(v, bool)
    )
    
    return comparison


def perform_injection_recovery_test(time, injection_params, n_trials=100, 
                                   noise_level=0.001, seed=42):
    """
    Perform injection-recovery test to assess detection efficiency.
    """
    np.random.seed(seed)
    
    recoveries = []
    recovered_params = []
    
    for i in range(n_trials):
        # Generate synthetic light curve
        clean_signal = generate_transit_signal_mandel_agol(
            time, 
            injection_params.period,
            injection_params.t0,
            np.sqrt(injection_params.depth),  # rprs
            10.0,  # aRs (placeholder)
            u1=0.1, u2=0.3
        )
        
        # Add noise
        noisy_flux = add_noise(clean_signal, noise_level=noise_level)
        
        # Try to recover
        try:
            result = find_transits_bls_advanced(
                time, noisy_flux, 
                min_period=injection_params.period * 0.9,
                max_period=injection_params.period * 1.1
            )
            
            recovered = result['period']
            recovery_success = (
                abs(recovered - injection_params.period) / injection_params.period < 0.01
            )
            
            recoveries.append(recovery_success)
            recovered_params.append({
                'period': recovered,
                'snr': result.get('snr', 0),
                'success': recovery_success
            })
        except:
            recoveries.append(False)
            recovered_params.append({'success': False})
    
    recovery_rate = np.mean(recoveries)
    
    return {
        'recovery_rate': recovery_rate,
        'n_trials': n_trials,
        'n_recovered': np.sum(recoveries),
        'injection_params': injection_params,
        'recovered_params': recovered_params,
        'detection_efficiency': recovery_rate
    }


def calculate_detection_significance(bls_result, n_shuffles=1000):
    """
    Calculate detection significance via data shuffling.
    """
    # Extract data from BLS result
    if isinstance(bls_result, dict):
        best_power = bls_result.get('power', 0)
        time = bls_result.get('time', None)
        flux = bls_result.get('flux', None)
    else:
        # Assume it's a BLS object
        best_power = np.max(bls_result.power.power)
        time = bls_result.t
        flux = bls_result.y
    
    if time is None or flux is None:
        return {'significance': 0, 'p_value': 1}
    
    # Shuffle data
    shuffled_powers = []
    
    for _ in range(n_shuffles):
        # Shuffle flux while preserving time
        shuffled_flux = np.random.permutation(flux)
        
        # Run quick BLS on shuffled data
        try:
            # Use a simplified BLS for speed
            from astropy.timeseries import BoxLeastSquares
            bls_shuffled = BoxLeastSquares(time, shuffled_flux)
            
            # Use a small period grid around interesting region
            periods = np.linspace(0.5, 20, 100)
            durations = np.array([0.05, 0.1, 0.2])
            
            power_shuffled = bls_shuffled.power(periods, durations)
            shuffled_powers.append(np.max(power_shuffled.power))
        except:
            shuffled_powers.append(0)
    
    # Calculate p-value
    shuffled_powers = np.array(shuffled_powers)
    p_value = np.sum(shuffled_powers >= best_power) / len(shuffled_powers)
    
    # Convert to sigma (Gaussian approximation)
    if p_value > 0:
        sigma = stats.norm.ppf(1 - p_value)
    else:
        sigma = np.inf
    
    return {
        'p_value': p_value,
        'significance_sigma': sigma,
        'best_power': best_power,
        'mean_shuffled_power': np.mean(shuffled_powers),
        'std_shuffled_power': np.std(shuffled_powers),
        'n_shuffles': n_shuffles
    }


def assess_multiple_event_statistic(time, flux, period, t0, duration):
    """
    Calculate Multiple Event Statistic (MES) for transit detection.
    Similar to Kepler pipeline.
    """
    # Phase fold
    phase = ((time - t0) / period) % 1
    phase = (phase + 0.5) % 1 - 0.5
    
    # Identify in-transit points
    half_width = 0.5 * duration / period
    in_transit = np.abs(phase) < half_width
    out_of_transit = ~in_transit
    
    if np.sum(in_transit) < 5:
        return {'mes': 0, 'n_transits': 0}
    
    # Calculate depth per transit
    transit_indices = np.floor((time[in_transit] - t0) / period + 0.5).astype(int)
    unique_transits = np.unique(transit_indices)
    
    depths = []
    for transit_n in unique_transits:
        # Points in this specific transit
        mask = transit_indices == transit_n
        if np.sum(mask) < 3:
            continue
        
        # Compare with nearby out-of-transit points
        transit_time = t0 + transit_n * period
        time_window = 2 * duration
        
        nearby_oot = (
            (time >= transit_time - time_window) & 
            (time <= transit_time + time_window) & 
            out_of_transit
        )
        
        if np.sum(nearby_oot) < 5:
            continue
        
        depth = np.median(flux[nearby_oot]) - np.median(flux[in_transit][mask])
        depths.append(depth)
    
    if len(depths) == 0:
        return {'mes': 0, 'n_transits': 0}
    
    # Calculate MES
    mean_depth = np.mean(depths)
    depth_uncertainty = np.std(depths) / np.sqrt(len(depths))
    
    mes = mean_depth / depth_uncertainty if depth_uncertainty > 0 else 0
    
    return {
        'mes': mes,
        'n_transits': len(depths),
        'mean_depth': mean_depth,
        'depth_uncertainty': depth_uncertainty,
        'individual_depths': depths
    }


def validate_against_secondary_eclipse(time, flux, period, t0, duration):
    """
    Check for secondary eclipse to validate planet detection.
    Secondary eclipse at phase 0.5 if eccentric.
    """
    # Look at phase 0.5 ± duration
    phase = ((time - t0) / period) % 1
    
    # Primary transit region
    primary_phase = 0.0
    primary_region = (phase > primary_phase - 0.5*duration/period) & \
                     (phase < primary_phase + 0.5*duration/period)
    
    # Secondary eclipse region
    secondary_phase = 0.5
    secondary_region = (phase > secondary_phase - 0.5*duration/period) & \
                       (phase < secondary_phase + 0.5*duration/period)
    
    # Compare depths
    if np.sum(primary_region) > 5 and np.sum(secondary_region) > 5:
        primary_depth = 1 - np.median(flux[primary_region])
        secondary_depth = 1 - np.median(flux[secondary_region])
        
        # Secondary should be much smaller or non-existent for non-irradiated planets
        secondary_ratio = secondary_depth / primary_depth if primary_depth > 0 else np.inf
        
        return {
            'primary_depth': primary_depth,
            'secondary_depth': secondary_depth,
            'secondary_ratio': secondary_ratio,
            'has_secondary': secondary_ratio > 0.1,  # Arbitrary threshold
            'n_primary': np.sum(primary_region),
            'n_secondary': np.sum(secondary_region)
        }
    
    return {
        'primary_depth': None,
        'secondary_depth': None,
        'secondary_ratio': None,
        'has_secondary': False,
        'n_primary': np.sum(primary_region),
        'n_secondary': np.sum(secondary_region)
    }