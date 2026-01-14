# analysis.py - Advanced statistical analysis
"""
Advanced statistical analysis for transit light curves.
Includes detrending, systematics removal, and significance testing.
"""

import numpy as np
from scipy import signal, optimize, stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings

def detrend_light_curve_gp(time, flux, flux_err=None, in_transit_mask=None,
                          kernel=None, optimize_hyperparams=True):
    """
    Detrend light curve using Gaussian Process regression.
    
    Parameters
    ----------
    time : array
        Time array
    flux : array
        Flux array
    flux_err : array, optional
        Flux errors
    in_transit_mask : array, optional
        Boolean mask of in-transit points to exclude from GP fit
    kernel : sklearn.gaussian_process.kernels.Kernel
        GP kernel
    optimize_hyperparams : bool
        Whether to optimize hyperparameters
        
    Returns
    -------
    detrended_flux : array
        Detrended flux
    trend : array
        Fitted trend
    gp : GaussianProcessRegressor
        Fitted GP model
    """
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.std(flux) / np.sqrt(len(flux))
    
    if in_transit_mask is None:
        # Fit to all data
        fit_mask = np.ones_like(flux, dtype=bool)
    else:
        # Exclude in-transit points
        fit_mask = ~in_transit_mask
    
    if kernel is None:
        # Default kernel: RBF + White noise
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    
    # Create GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=flux_err[fit_mask]**2,
        normalize_y=True,
        n_restarts_optimizer=10 if optimize_hyperparams else 0
    )
    
    # Reshape for sklearn
    X = time[fit_mask].reshape(-1, 1)
    y = flux[fit_mask]
    
    # Fit GP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    
    # Predict trend
    X_all = time.reshape(-1, 1)
    trend, trend_std = gp.predict(X_all, return_std=True)
    
    # Detrend
    detrended_flux = flux - trend + np.median(trend)
    
    return detrended_flux, trend, gp


def remove_systematics_pca(time, flux, n_components=5):
    """
    Remove systematics using Principal Component Analysis.
    Useful for TESS/Kepler data with common systematics.
    """
    from sklearn.decomposition import PCA
    
    # Create design matrix (time derivatives, etc.)
    # In practice, you'd use co-trending basis vectors
    n_points = len(time)
    
    # Simple polynomial basis
    X = np.column_stack([
        np.ones(n_points),
        time,
        time**2,
        time**3,
        np.sin(2*np.pi*time),
        np.cos(2*np.pi*time)
    ])
    
    # Add more features based on known systematics
    # ...
    
    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    
    # Regress out components
    coeffs = np.linalg.lstsq(components, flux, rcond=None)[0]
    systematics = components @ coeffs
    
    corrected_flux = flux - systematics + np.mean(systematics)
    
    return {
        'corrected_flux': corrected_flux,
        'systematics': systematics,
        'components': components,
        'explained_variance': pca.explained_variance_ratio_,
        'pca': pca
    }


def correct_for_airmass(time, flux, airmass, flux_err=None):
    """
    Correct for airmass effects (ground-based observations).
    """
    # Model airmass effect as exponential
    def airmass_model(params, am):
        a, b = params
        return a * np.exp(b * (am - 1))
    
    def residuals(params, am, f):
        return f - airmass_model(params, am)
    
    # Fit airmass correction
    params0 = [0.01, 0.2]
    result = optimize.least_squares(residuals, params0, args=(airmass, flux))
    
    correction = airmass_model(result.x, airmass)
    corrected_flux = flux / (1 + correction)
    
    return {
        'corrected_flux': corrected_flux,
        'correction': correction,
        'params': result.x,
        'success': result.success
    }


def measure_transit_timing_variations(time, flux, period, t0, duration,
                                     epoch_window=5):
    """
    Measure Transit Timing Variations (TTVs).
    
    Parameters
    ----------
    epoch_window : int
        Number of epochs to include in TTV measurement
        
    Returns
    -------
    ttv_results : dict
        TTV measurements and significance
    """
    # Predict transit centers
    tmin, tmax = time.min(), time.max()
    n_min = int(np.floor((tmin - t0) / period)) - 1
    n_max = int(np.ceil((tmax - t0) / period)) + 1
    
    ttv_measurements = []
    
    for n in range(n_min, n_max + 1):
        tc_pred = t0 + n * period
        
        # Extract window around predicted transit
        window = 3 * duration
        mask = (time >= tc_pred - window) & (time <= tc_pred + window)
        
        if np.sum(mask) < 20:
            continue
        
        t_window = time[mask]
        f_window = flux[mask]
        
        # Fit transit time
        try:
            tc_measured, tc_err = fit_transit_time(
                t_window, f_window, period, tc_pred, duration
            )
            
            ttv = tc_measured - tc_pred
            
            ttv_measurements.append({
                'epoch': n,
                'tc_pred': tc_pred,
                'tc_measured': tc_measured,
                'tc_err': tc_err,
                'ttv': ttv,
                'ttv_sigma': ttv / tc_err if tc_err > 0 else np.inf,
                'n_points': len(t_window)
            })
        except:
            continue
    
    # Analyze TTVs
    if len(ttv_measurements) < 3:
        return {'ttvs_detected': False}
    
    epochs = np.array([m['epoch'] for m in ttv_measurements])
    ttvs = np.array([m['ttv'] for m in ttv_measurements])
    ttv_errs = np.array([m['tc_err'] for m in ttv_measurements])
    
    # Check for significant TTVs
    chi2 = np.sum((ttvs / ttv_errs) ** 2)
    dof = len(ttvs) - 1
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    # Fit sinusoidal TTV (potential planet-planet interactions)
    if len(ttvs) >= 10:
        try:
            ttv_period, ttv_amplitude = fit_sinusoidal_ttv(epochs, ttvs, ttv_errs)
        except:
            ttv_period = ttv_amplitude = np.nan
    
    return {
        'ttvs_detected': p_value < 0.01,
        'p_value': p_value,
        'chi2': chi2,
        'dof': dof,
        'measurements': ttv_measurements,
        'epochs': epochs,
        'ttvs': ttvs,
        'ttv_errs': ttv_errs,
        'rms_ttv': np.sqrt(np.mean(ttvs**2)),
        'ttv_period': ttv_period if 'ttv_period' in locals() else np.nan,
        'ttv_amplitude': ttv_amplitude if 'ttv_amplitude' in locals() else np.nan
    }


def fit_transit_time(time, flux, period, t0_guess, duration):
    """Fit precise transit time for a single epoch."""
    # Define transit model
    def transit_model(params, t):
        t0, depth = params
        phase = ((t - t0) / period) % 1
        half_width = 0.5 * duration / period
        
        model = np.ones_like(t)
        in_transit = (phase < half_width) | (phase > 1 - half_width)
        model[in_transit] = 1 - depth
        
        return model
    
    def residuals(params, t, f):
        return f - transit_model(params, t)
    
    # Initial guess
    params0 = [t0_guess, 0.01]
    
    # Fit
    result = optimize.least_squares(residuals, params0, args=(time, flux))
    
    # Estimate errors from covariance matrix
    jac = result.jac
    try:
        cov = np.linalg.inv(jac.T @ jac)
        errors = np.sqrt(np.diag(cov))
        t0_err = errors[0]
    except:
        t0_err = duration / 10  # Rough estimate
    
    return result.x[0], t0_err


def fit_sinusoidal_ttv(epochs, ttvs, ttv_errs):
    """Fit sinusoidal TTV pattern."""
    def sinusoid(params, t):
        A, P, phi, offset = params
        return A * np.sin(2*np.pi*t/P + phi) + offset
    
    def residuals(params, t, y, yerr):
        return (y - sinusoid(params, t)) / yerr
    
    # Initial guesses
    A_guess = np.std(ttvs)
    P_guess = len(ttvs) / 2  # Rough guess
    params0 = [A_guess, P_guess, 0, np.mean(ttvs)]
    
    # Fit
    bounds = ([0, 1, -np.pi, -np.inf], 
              [np.inf, len(ttvs)*2, np.pi, np.inf])
    
    result = optimize.least_squares(residuals, params0, 
                                    args=(epochs, ttvs, ttv_errs),
                                    bounds=bounds)
    
    return result.x[1], result.x[0]  # Period, Amplitude


def calculate_transit_duration_ratio(primary_duration, secondary_duration=None,
                                   period=None, eccentricity=0, omega=90):
    """
    Calculate transit duration ratio for eccentricity constraints.
    
    For eccentric orbits, duration varies with true anomaly.
    """
    if secondary_duration is None:
        # Need other parameters
        return None
    
    ratio = primary_duration / secondary_duration
    
    # Expected ratio for circular orbit
    expected_circular = 1.0
    
    # Correction for eccentricity
    if eccentricity > 0 and period is not None:
        # Simplified correction (Winn 2010)
        omega_rad = np.radians(omega)
        factor = (1 + eccentricity * np.sin(omega_rad)) / np.sqrt(1 - eccentricity**2)
        expected_eccentric = factor
    else:
        expected_eccentric = 1.0
    
    return {
        'measured_ratio': ratio,
        'expected_circular': expected_circular,
        'expected_eccentric': expected_eccentric,
        'eccentricity_suggested': abs(ratio - 1) > 0.1
    }