# utils.py - Enhanced utilities
"""
Utilities for transit analysis.
"""

import numpy as np
from scipy import signal, stats
from astropy import units as u
from astropy.time import Time
import warnings

def calculate_snr(time, flux, period, t0, duration):
    """
    Calculate transit signal-to-noise ratio.
    """
    # Phase fold
    phase = ((time - t0) / period) % 1
    phase = (phase + 0.5) % 1 - 0.5
    
    # Identify in-transit points
    half_width = 0.5 * duration / period
    in_transit = np.abs(phase) < half_width
    out_of_transit = ~in_transit
    
    if np.sum(in_transit) < 10 or np.sum(out_of_transit) < 10:
        return 0.0
    
    # Calculate depths
    depth = np.median(flux[out_of_transit]) - np.median(flux[in_transit])
    
    # Noise from out-of-transit RMS
    noise = np.std(flux[out_of_transit]) / np.sqrt(np.sum(in_transit))
    
    snr = depth / noise if noise > 0 else 0.0
    
    return snr


def estimate_limb_darkening(teff, logg, feh, method='claret'):
    """
    Estimate limb darkening coefficients.
    
    Methods:
    - 'claret': Claret & Bloemen (2011) tables
    - 'quadratic': Generic quadratic coefficients
    - 'linear': Generic linear coefficient
    """
    if method == 'claret':
        # Simplified approximation
        # In practice, use Claret tables or LDCU
        if teff < 4000:  # M-dwarfs
            u1, u2 = 0.5, 0.3
        elif teff < 5500:  # K-dwarfs
            u1, u2 = 0.4, 0.3
        elif teff < 6500:  # G-dwarfs
            u1, u2 = 0.3, 0.3
        elif teff < 8000:  # F-dwarfs
            u1, u2 = 0.2, 0.3
        else:  # A-dwarfs and hotter
            u1, u2 = 0.1, 0.3
    elif method == 'quadratic':
        u1, u2 = 0.3, 0.3
    elif method == 'linear':
        u1, u2 = 0.6, 0.0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return u1, u2


def calculate_transit_duration_from_parameters(period, aRs, rprs, b=0):
    """
    Calculate transit duration from orbital parameters.
    
    Using Winn (2010) Equation 14.
    """
    # Impact parameter correction
    if b >= 1:
        return 0.0
    
    # Circular orbit
    duration = period / np.pi * np.arcsin(
        np.sqrt((1 + rprs)**2 - b**2) / aRs
    )
    
    return duration


def calculate_scaled_semi_major_axis(period, stellar_mass, stellar_radius):
    """
    Calculate a/R_star from stellar parameters.
    
    Using Kepler's third law.
    """
    # Convert to SI
    period_sec = period * 24 * 3600  # days to seconds
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M_si = stellar_mass * 1.989e30  # kg
    R_si = stellar_radius * 6.957e8  # m
    
    # Semi-major axis
    a = (G * M_si * period_sec**2 / (4 * np.pi**2)) ** (1/3)
    
    # Scaled
    aRs = a / R_si
    
    return aRs


def estimate_planet_radius(depth, stellar_radius):
    """
    Estimate planet radius from transit depth.
    
    R_planet = R_star * sqrt(depth)
    """
    rprs = np.sqrt(depth)
    planet_radius = stellar_radius * rprs
    
    return planet_radius, rprs


def calculate_transit_probability(rprs, aRs):
    """
    Calculate geometric transit probability.
    
    P_transit = (R_star + R_planet) / a
    """
    probability = (1 + rprs) / aRs
    
    return min(probability, 1.0)


def time_to_phase(time, period, t0):
    """Convert time to phase."""
    phase = ((time - t0) / period) % 1
    return phase


def phase_to_time(phase, period, t0, epoch=0):
    """Convert phase to time."""
    time = t0 + period * (epoch + phase)
    return time


def calculate_rms(time, flux, bin_size=30):
    """
    Calculate RMS vs bin size (Allan variance).
    """
    n_bins = len(time) // bin_size
    if n_bins < 2:
        return np.nan, np.nan
    
    binned_flux = np.array_split(flux[:n_bins*bin_size], n_bins)
    binned_means = [np.mean(bin_data) for bin_data in binned_flux]
    
    rms = np.std(binned_means)
    expected_rms = np.std(flux) / np.sqrt(bin_size)
    
    return rms, expected_rms


def detect_outliers_modified_zscore(data, threshold=3.5):
    """
    Detect outliers using modified Z-score method.
    More robust than standard deviation.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        # Use standard deviation if MAD is zero
        mad = 1.4826 * np.std(data)
    
    modified_z_scores = 0.6745 * (data - median) / mad
    
    return np.abs(modified_z_scores) > threshold


def filter_lowpass(data, cutoff_freq, sample_rate, order=5):
    """
    Apply low-pass Butterworth filter.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, data)
    
    return filtered


def calculate_cdpp(flux, window_length=13):
    """
    Calculate Combined Differential Photometric Precision (CDPP).
    Used for Kepler/TESS noise metric.
    """
    # Remove trends
    detrended = signal.detrend(flux)
    
    # Calculate RMS in rolling windows
    n = len(detrended)
    cdpp_values = []
    
    for i in range(0, n - window_length + 1, window_length):
        window = detrended[i:i+window_length]
        cdpp_values.append(np.std(window))
    
    # Median CDPP in ppm
    cdpp_ppm = np.median(cdpp_values) * 1e6
    
    return cdpp_ppm


def calculate_phase_coverage(time, period, t0):
    """
    Calculate phase coverage of observations.
    """
    phases = time_to_phase(time, period, t0)
    
    # Bin phases
    phase_bins = np.linspace(0, 1, 101)
    phase_counts, _ = np.histogram(phases, bins=phase_bins)
    
    coverage = np.mean(phase_counts > 0)
    
    return coverage


def check_data_quality(time, flux, flux_err=None):
    """
    Perform basic data quality checks.
    """
    quality = {}
    
    # Check for NaNs
    quality['n_nans'] = np.sum(~np.isfinite(flux))
    quality['has_nans'] = quality['n_nans'] > 0
    
    # Check time sorting
    quality['is_sorted'] = np.all(np.diff(time) >= 0)
    
    # Check time gaps
    time_gaps = np.diff(time)
    quality['max_gap'] = np.max(time_gaps) if len(time_gaps) > 0 else 0
    quality['median_gap'] = np.median(time_gaps) if len(time_gaps) > 0 else 0
    
    # Check flux statistics
    quality['flux_mean'] = np.nanmean(flux)
    quality['flux_std'] = np.nanstd(flux)
    quality['flux_median'] = np.nanmedian(flux)
    
    # Check for large outliers
    if flux_err is not None:
        residuals = flux - quality['flux_median']
        sigmas = np.abs(residuals) / flux_err
        quality['n_sigma_5'] = np.sum(sigmas > 5)
        quality['n_sigma_10'] = np.sum(sigmas > 10)
    
    # Data span
    quality['time_span'] = time[-1] - time[0] if len(time) > 1 else 0
    quality['n_points'] = len(time)
    
    # Calculate CDPP-like metric
    quality['noise_ppm'] = calculate_cdpp(flux) if len(flux) > 100 else np.nan
    
    return quality