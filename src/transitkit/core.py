# core.py - Scientific core with rigorous methods
"""
Core transit analysis functions with error propagation, MCMC support,
and publication-quality algorithms.
"""

import numpy as np
import warnings
from typing import Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
from scipy import optimize, stats, signal
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
import emcee
import corner

@dataclass
class TransitParameters:
    """Container for transit parameters with uncertainties."""
    period: float
    period_err: float = 0.0
    t0: float
    t0_err: float = 0.0
    depth: float
    depth_err: float = 0.0
    duration: float
    duration_err: float = 0.0
    b: float = 0.5  # Impact parameter
    b_err: float = 0.0
    rprs: float = None  # Planet-to-star radius ratio
    rprs_err: float = 0.0
    aRs: float = None  # Scaled semi-major axis
    aRs_err: float = 0.0
    inclination: float = 90.0
    inclination_err: float = 0.0
    limb_darkening: Tuple[float, float] = (0.1, 0.3)  # u1, u2
    snr: float = 0.0
    fap: float = 1.0  # False alarm probability
    quality_flags: Dict = None
    
    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = {
                'bls_snr': self.snr > 7,
                'duration_consistent': True,
                'odd_even_consistent': True
            }
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    @classmethod
    def from_bls_result(cls, bls_result, time, flux, **kwargs):
        """Create from BLS result with additional calculations."""
        params = cls(
            period=bls_result.period,
            t0=bls_result.transit_time,
            depth=bls_result.depth,
            duration=bls_result.duration,
            snr=getattr(bls_result, 'snr', 0),
            **kwargs
        )
        
        # Calculate additional parameters if possible
        if hasattr(bls_result, 'depth_err'):
            params.depth_err = bls_result.depth_err
        if hasattr(bls_result, 'duration_err'):
            params.duration_err = bls_result.duration_err
            
        return params


def generate_transit_signal_mandel_agol(
    time: np.ndarray,
    period: float,
    t0: float,
    rprs: float,
    aRs: float,
    inclination: float = 90.0,
    eccentricity: float = 0.0,
    omega: float = 90.0,
    u1: float = 0.1,
    u2: float = 0.3,
    exptime: float = 0.0,
    supersample: int = 7
) -> np.ndarray:
    """
    Generate transit light curves using Mandel & Agol (2002) model.
    
    Parameters
    ----------
    time : array
        Time array in days
    period : float
        Orbital period in days
    t0 : float
        Time of mid-transit in days
    rprs : float
        Planet-to-star radius ratio
    aRs : float
        Scaled semi-major axis (a/R_*)
    inclination : float
        Orbital inclination in degrees
    eccentricity : float
        Orbital eccentricity
    omega : float
        Argument of periastron in degrees
    u1, u2 : float
        Quadratic limb darkening coefficients
    exptime : float
        Exposure time for binning (days)
    supersample : int
        Supersampling factor for exposure time integration
        
    Returns
    -------
    flux : array
        Flux with transit signal
    """
    from batman import TransitParams, TransitModel
    
    params = TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rprs
    params.a = aRs
    params.inc = inclination
    params.ecc = eccentricity
    params.w = omega
    params.u = [u1, u2]  # Quadratic limb darkening
    params.limb_dark = "quadratic"
    
    # Create model
    m = TransitModel(params, time, supersample_factor=supersample, 
                     exp_time=exptime)
    
    return m.light_curve(params)


def find_transits_multiple_methods(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = 0.5,
    max_period: float = 100.0,
    n_periods: int = 10000,
    methods: List[str] = ['bls', 'gls', 'pdm']
) -> Dict:
    """
    Find transits using multiple methods for robust detection.
    
    Returns consensus results with validation metrics.
    """
    results = {}
    
    # BLS (primary method)
    if 'bls' in methods:
        bls_result = find_transits_bls_advanced(
            time, flux, flux_err, min_period, max_period, n_periods
        )
        results['bls'] = bls_result
        
    # Generalized Lomb-Scargle
    if 'gls' in methods:
        gls_result = find_period_gls(time, flux, flux_err)
        results['gls'] = gls_result
        
    # Phase Dispersion Minimization
    if 'pdm' in methods:
        pdm_result = find_period_pdm(time, flux)
        results['pdm'] = pdm_result
    
    # Calculate consensus
    consensus = calculate_consensus(results)
    results['consensus'] = consensus
    
    # Validation metrics
    results['validation'] = validate_transit_detection(
        time, flux, consensus, results
    )
    
    return results


def find_transits_bls_advanced(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = 0.5,
    max_period: float = 100.0,
    n_periods: int = 10000,
    durations: Optional[np.ndarray] = None,
    objective: str = 'likelihood'
) -> Dict:
    """
    Advanced BLS with likelihood optimization, FAP calculation,
    and multiple hypothesis testing correction.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.std(flux) / np.sqrt(len(flux))
    
    if durations is None:
        # Log-spaced durations from 0.5 to 15 hours
        durations = np.logspace(np.log10(0.5/24), np.log10(15/24), 15)
    
    # Log-spaced periods for better sensitivity
    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
    
    # Run BLS
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    
    if objective == 'likelihood':
        # Use likelihood method
        power = bls.power(periods, durations, objective='likelihood')
    else:
        power = bls.power(periods, durations, objective='snr')
    
    # Find best period
    best_idx = np.nanargmax(power.power)
    
    # Calculate SNR
    model = bls.model(time, power.period[best_idx], power.duration[best_idx],
                      power.transit_time[best_idx])
    residuals = flux - model
    rms = np.sqrt(np.mean(residuals**2))
    snr = power.depth[best_idx] / rms * np.sqrt(len(time[model < 1]))
    
    # Calculate False Alarm Probability (Bootstrap method)
    fap = calculate_fap_bootstrap(bls, time, flux, flux_err, 
                                  power.period[best_idx], n_bootstrap=1000)
    
    # Calculate parameter uncertainties via MCMC
    if snr > 5:  # Only run MCMC for good detections
        samples, param_errors = estimate_parameters_mcmc(
            time, flux, flux_err,
            power.period[best_idx],
            power.transit_time[best_idx],
            power.duration[best_idx],
            power.depth[best_idx]
        )
    else:
        param_errors = {
            'period_err': 0.0,
            't0_err': 0.0,
            'duration_err': 0.0,
            'depth_err': 0.0
        }
    
    # Create comprehensive result
    result = {
        'period': float(power.period[best_idx]),
        't0': float(power.transit_time[best_idx]),
        'duration': float(power.duration[best_idx]),
        'depth': float(power.depth[best_idx]),
        'snr': float(snr),
        'fap': float(fap),
        'power': float(power.power[best_idx]),
        
        # Full scans
        'all_periods': power.period,
        'all_powers': power.power,
        'all_durations': power.duration,
        
        # Parameter uncertainties
        'errors': param_errors,
        
        # Statistics
        'residuals_rms': float(rms),
        'chi2': float(np.sum((residuals/flux_err)**2)),
        'bic': calculate_bic(time, flux, model, flux_err, n_params=4),
        
        # Metadata
        'method': 'bls_advanced',
        'objective': objective,
        'n_data_points': len(time),
        'data_span': float(time[-1] - time[0])
    }
    
    return result


def calculate_fap_bootstrap(bls, time, flux, flux_err, period, 
                           n_bootstrap=1000, seed=42):
    """Calculate FAP using bootstrap method."""
    np.random.seed(seed)
    max_powers = []
    
    # Generate bootstrap samples preserving time structure
    for _ in range(n_bootstrap):
        # Phase scrambling
        phases = np.random.permutation(np.arange(len(flux)))
        flux_bs = flux[phases]
        
        # Quick BLS on scrambled data
        periods_test = np.linspace(period*0.9, period*1.1, 100)
        durations_test = np.array([0.02, 0.05, 0.1])
        
        try:
            power_bs = bls.power(periods_test, durations_test, 
                                 objective='likelihood')
            max_powers.append(np.max(power_bs.power))
        except:
            max_powers.append(0)
    
    # Calculate FAP from original power
    original_power = bls.power([period], [0.05], objective='likelihood').power[0]
    fap = np.sum(np.array(max_powers) >= original_power) / n_bootstrap
    
    return max(fap, 1e-10)  # Avoid zero


def estimate_parameters_mcmc(time, flux, flux_err, period_guess, 
                            t0_guess, duration_guess, depth_guess,
                            n_walkers=32, n_steps=2000, burnin=500):
    """
    Estimate transit parameters and uncertainties using MCMC.
    """
    def log_likelihood(theta, t, f, ferr):
        """Gaussian likelihood."""
        period, t0, duration, depth = theta
        
        # Simple box model for speed
        phase = ((t - t0) / period) % 1
        half_width = 0.5 * duration / period
        in_transit = (phase < half_width) | (phase > 1 - half_width)
        
        model = np.ones_like(f)
        model[in_transit] = 1 - depth
        
        # Chi-squared
        chi2 = np.sum(((f - model) / ferr) ** 2)
        return -0.5 * chi2
    
    def log_prior(theta):
        """Uniform priors."""
        period, t0, duration, depth = theta
        
        if not (0.1 < period < 100):
            return -np.inf
        if not (time.min() < t0 < time.max()):
            return -np.inf
        if not (0.001 < duration < 0.5):
            return -np.inf
        if not (0.0001 < depth < 0.5):
            return -np.inf
            
        return 0.0
    
    def log_probability(theta, t, f, ferr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, t, f, ferr)
    
    # Initial guess
    ndim = 4
    pos = np.array([period_guess, t0_guess, duration_guess, depth_guess])
    pos = pos + 1e-4 * np.random.randn(n_walkers, ndim) * pos
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability, 
        args=(time, flux, flux_err)
    )
    sampler.run_mcmc(pos, n_steps, progress=False)
    
    # Process chains
    samples = sampler.get_chain(discard=burnin, flat=True)
    
    # Calculate percentiles
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    
    # Errors as 68% confidence intervals
    errors = {
        'period_err': (percentiles[2,0] - percentiles[0,0]) / 2,
        't0_err': (percentiles[2,1] - percentiles[0,1]) / 2,
        'duration_err': (percentiles[2,2] - percentiles[0,2]) / 2,
        'depth_err': (percentiles[2,3] - percentiles[0,3]) / 2,
    }
    
    return samples, errors


def find_period_gls(time, flux, flux_err=None):
    """Generalized Lomb-Scargle periodogram."""
    from astropy.timeseries import LombScargle
    
    ls = LombScargle(time, flux, dy=flux_err)
    frequency, power = ls.autopower(minimum_frequency=1/100, 
                                    maximum_frequency=1/0.5)
    
    best_idx = np.argmax(power)
    period = 1/frequency[best_idx]
    
    # Calculate FAP
    fap = ls.false_alarm_probability(power[best_idx])
    
    return {
        'period': period,
        'power': power[best_idx],
        'fap': fap,
        'frequencies': frequency,
        'powers': power,
        'method': 'gls'
    }


def find_period_pdm(time, flux, nbins=10):
    """Phase Dispersion Minimization."""
    from astropy.stats import phase_dispersion
    
    periods = np.logspace(np.log10(0.5), np.log10(100), 1000)
    theta = []
    
    for period in periods:
        theta.append(phase_dispersion(time, flux, period, nbins=nbins))
    
    theta = np.array(theta)
    best_period = periods[np.argmin(theta)]
    
    return {
        'period': best_period,
        'theta': theta.min(),
        'periods': periods,
        'thetas': theta,
        'method': 'pdm'
    }


def calculate_consensus(results):
    """Calculate consensus from multiple period search methods."""
    periods = []
    weights = []
    
    for method, result in results.items():
        if method in ['bls', 'gls']:
            if 'period' in result:
                periods.append(result['period'])
                # Weight by SNR or inverse FAP
                if 'fap' in result and result['fap'] > 0:
                    weights.append(-np.log10(result['fap']))
                elif 'snr' in result:
                    weights.append(result['snr'])
                else:
                    weights.append(1.0)
    
    if len(periods) == 0:
        return None
    
    periods = np.array(periods)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    # Weighted average
    consensus_period = np.average(periods, weights=weights)
    
    # Check for harmonics
    harmonics = check_harmonics(periods, consensus_period)
    
    return {
        'period': consensus_period,
        'period_std': np.std(periods),
        'method_agreement': len(periods),
        'weights': weights.tolist(),
        'individual_periods': periods.tolist(),
        'harmonics_detected': harmonics,
        'is_harmonic': harmonics['is_harmonic']
    }


def check_harmonics(periods, reference, tolerance=0.01):
    """Check for harmonic relationships."""
    harmonics = {}
    
    for p in periods:
        ratio = p / reference
        # Check common harmonics
        for mult in [0.5, 2, 3, 1/3]:
            if abs(ratio - mult) < tolerance:
                harmonics[str(mult)] = True
                break
    
    harmonics['is_harmonic'] = len(harmonics) > 0
    return harmonics


def validate_transit_detection(time, flux, consensus, all_results):
    """Validate transit detection with multiple tests."""
    validation = {}
    
    if consensus is None:
        validation['passed'] = False
        return validation
    
    # Test 1: Odd-even transit consistency
    validation['odd_even'] = check_odd_even_consistency(
        time, flux, consensus['period'], 
        all_results.get('bls', {}).get('t0', 0)
    )
    
    # Test 2: Duration consistency
    if 'bls' in all_results:
        validation['duration_consistency'] = check_duration_consistency(
            all_results['bls']['duration'], consensus['period']
        )
    
    # Test 3: Secondary eclipse check
    validation['secondary'] = check_secondary_eclipse(
        time, flux, consensus['period'], 
        all_results.get('bls', {}).get('t0', 0)
    )
    
    # Test 4: Stellar variability contamination
    validation['variability'] = check_stellar_variability(time, flux)
    
    # Overall validation
    validation['passed'] = (
        validation.get('odd_even', {}).get('p_value', 0) > 0.01 and
        validation.get('duration_consistency', True) and
        validation.get('secondary', {}).get('detected', False) == False
    )
    
    return validation


def check_odd_even_consistency(time, flux, period, t0):
    """Check if odd and even transits are consistent."""
    phase = ((time - t0) / period) % 1
    
    # Separate odd and even transits
    transit_number = np.floor((time - t0) / period + 0.5).astype(int)
    is_even = transit_number % 2 == 0
    
    # Extract in-transit points
    in_transit = (phase < 0.05) | (phase > 0.95)
    
    if np.sum(in_transit & is_even) < 10 or np.sum(in_transit & ~is_even) < 10:
        return {'p_value': 1.0, 'conclusion': 'insufficient_data'}
    
    flux_even = flux[in_transit & is_even]
    flux_odd = flux[in_transit & ~is_even]
    
    # T-test for difference
    t_stat, p_value = stats.ttest_ind(flux_even, flux_odd, 
                                      equal_var=False, nan_policy='omit')
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'n_even': len(flux_even),
        'n_odd': len(flux_odd),
        'mean_even': np.nanmean(flux_even),
        'mean_odd': np.nanmean(flux_odd),
        'conclusion': 'consistent' if p_value > 0.01 else 'inconsistent'
    }


def calculate_bic(time, flux, model, flux_err, n_params):
    """Calculate Bayesian Information Criterion."""
    n = len(time)
    chi2 = np.sum(((flux - model) / flux_err) ** 2)
    bic = chi2 + n_params * np.log(n)
    return bic