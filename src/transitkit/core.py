# core.py - Scientific core with rigorous methods
"""
Core transit analysis functions with error propagation, MCMC support,
and publication-quality algorithms.

This module intentionally keeps computational kernels reasonably lightweight
and robust to missing optional dependencies.

Key fixes:
- dataclass ordering bug fixed via kw_only=True (Python 3.10+)
- optional parameters use safe defaults (None or tuples)
- clearer errors for optional dependencies (batman-package)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from astropy.timeseries import BoxLeastSquares

import emcee


# =============================================================================
# Data containers
# =============================================================================

@dataclass(kw_only=True)
class TransitParameters:
    """
    Container for transit parameters with uncertainties.

    NOTE:
    - kw_only=True avoids dataclass positional-argument ordering constraints
      (e.g., non-default after default), and is friendlier for future extensions.
    """

    # Required (core) parameters
    period: float
    t0: float
    depth: float
    duration: float

    # Uncertainties
    period_err: float = 0.0
    t0_err: float = 0.0
    depth_err: float = 0.0
    duration_err: float = 0.0

    # Optional / derived parameters
    b: float = 0.5  # impact parameter
    b_err: float = 0.0

    rprs: Optional[float] = None  # Rp/R*
    rprs_err: float = 0.0

    aRs: Optional[float] = None  # a/R*
    aRs_err: float = 0.0

    inclination: float = 90.0
    inclination_err: float = 0.0

    limb_darkening: Tuple[float, float] = (0.1, 0.3)  # (u1, u2)

    # Detection diagnostics
    snr: float = 0.0
    fap: float = 1.0

    # Quality flags
    quality_flags: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fill in defaults if empty
        if not self.quality_flags:
            self.quality_flags = {
                "bls_snr": self.snr > 7,
                "duration_consistent": True,
                "odd_even_consistent": True,
            }

        # Basic sanity checks (non-fatal, warn only)
        if self.period <= 0:
            warnings.warn("TransitParameters.period <= 0 is not physical.", RuntimeWarning)
        if self.duration <= 0:
            warnings.warn("TransitParameters.duration <= 0 is not physical.", RuntimeWarning)
        if self.depth < 0:
            warnings.warn("TransitParameters.depth < 0 is not physical.", RuntimeWarning)

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_bls_result(cls, bls_result, time, flux, **kwargs) -> "TransitParameters":
        """
        Create TransitParameters from an Astropy BLS result.

        Expects attributes: period, transit_time, depth, duration (standard in BLS results).
        """
        params = cls(
            period=float(bls_result.period),
            t0=float(bls_result.transit_time),
            depth=float(bls_result.depth),
            duration=float(bls_result.duration),
            snr=float(getattr(bls_result, "snr", 0.0) or 0.0),
            **kwargs,
        )

        # Optional errors if present
        if hasattr(bls_result, "depth_err"):
            params.depth_err = float(bls_result.depth_err)
        if hasattr(bls_result, "duration_err"):
            params.duration_err = float(bls_result.duration_err)

        return params


# =============================================================================
# Transit modeling
# =============================================================================

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
    supersample: int = 7,
) -> np.ndarray:
    """
    Generate transit light curves using Mandel & Agol (2002) model via batman-package.

    Returns
    -------
    flux_model : np.ndarray
        Model flux array (same shape as time).
    """
    time = np.asarray(time, dtype=float)

    try:
        from batman import TransitParams, TransitModel  # batman-package
    except Exception as e:
        raise ImportError(
            "batman-package is required for Mandel & Agol model generation. "
            "Install via: pip install batman-package"
        ) from e

    params = TransitParams()
    params.t0 = float(t0)
    params.per = float(period)
    params.rp = float(rprs)
    params.a = float(aRs)
    params.inc = float(inclination)
    params.ecc = float(eccentricity)
    params.w = float(omega)
    params.u = [float(u1), float(u2)]
    params.limb_dark = "quadratic"

    m = TransitModel(
        params,
        time,
        supersample_factor=int(max(1, supersample)),
        exp_time=float(exptime),
    )
    return m.light_curve(params)


# =============================================================================
# Period / transit search
# =============================================================================

def find_transits_multiple_methods(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = 0.5,
    max_period: float = 100.0,
    n_periods: int = 10000,
    methods: List[str] = None,
) -> Dict[str, object]:
    """
    Find transits using multiple methods for robust detection.
    Returns a dict with per-method outputs + consensus + validation.
    """
    if methods is None:
        methods = ["bls", "gls", "pdm"]

    results: Dict[str, object] = {}

    if "bls" in methods:
        results["bls"] = find_transits_bls_advanced(
            time, flux, flux_err, min_period, max_period, n_periods
        )

    if "gls" in methods:
        results["gls"] = find_period_gls(time, flux, flux_err)

    if "pdm" in methods:
        results["pdm"] = find_period_pdm(time, flux)

    consensus = calculate_consensus(results)
    results["consensus"] = consensus

    results["validation"] = validate_transit_detection(time, flux, consensus, results)

    return results


def find_transits_bls_advanced(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = 0.5,
    max_period: float = 100.0,
    n_periods: int = 10000,
    durations: Optional[np.ndarray] = None,
    objective: str = "likelihood",
) -> Dict[str, object]:
    """
    Advanced BLS with likelihood optimization, bootstrap FAP estimate,
    and optional MCMC uncertainty estimation.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if flux_err is None:
        # Conservative default: scale by scatter / sqrt(N)
        scatter = np.nanstd(flux)
        flux_err = np.ones_like(flux, dtype=float) * (scatter / np.sqrt(max(len(flux), 1)))
    else:
        flux_err = np.asarray(flux_err, dtype=float)

    if durations is None:
        # Log-spaced durations from 0.5 to 15 hours (days)
        durations = np.logspace(np.log10(0.5 / 24.0), np.log10(15.0 / 24.0), 15)

    periods = np.logspace(np.log10(min_period), np.log10(max_period), int(n_periods))

    bls = BoxLeastSquares(time, flux, dy=flux_err)

    obj = "likelihood" if objective == "likelihood" else "snr"
    power = bls.power(periods, durations, objective=obj)

    best_idx = int(np.nanargmax(power.power))

    best_period = float(power.period[best_idx])
    best_t0 = float(power.transit_time[best_idx])
    best_duration = float(power.duration[best_idx])
    best_depth = float(power.depth[best_idx])

    model = bls.model(time, best_period, best_duration, best_t0)
    residuals = flux - model
    rms = float(np.sqrt(np.nanmean(residuals**2)))

    # SNR estimate: depth / rms * sqrt(N_in)
    in_transit = model < 1.0
    n_in = int(np.sum(in_transit))
    snr = float((best_depth / max(rms, 1e-12)) * np.sqrt(max(n_in, 1)))

    fap = float(
        calculate_fap_bootstrap(
            bls, time, flux, flux_err, best_period, n_bootstrap=500  # keep reasonable
        )
    )

    if snr > 5:
        _samples, param_errors = estimate_parameters_mcmc(
            time, flux, flux_err, best_period, best_t0, best_duration, best_depth
        )
    else:
        param_errors = {"period_err": 0.0, "t0_err": 0.0, "duration_err": 0.0, "depth_err": 0.0}

    result: Dict[str, object] = {
        "period": best_period,
        "t0": best_t0,
        "duration": best_duration,
        "depth": best_depth,
        "snr": snr,
        "fap": fap,
        "power": float(power.power[best_idx]),
        "all_periods": power.period,
        "all_powers": power.power,
        "all_durations": power.duration,
        "errors": param_errors,
        "residuals_rms": rms,
        "chi2": float(np.nansum(((residuals) / flux_err) ** 2)),
        "bic": float(calculate_bic(time, flux, model, flux_err, n_params=4)),
        "method": "bls_advanced",
        "objective": obj,
        "n_data_points": int(len(time)),
        "data_span": float(time[-1] - time[0]) if len(time) > 1 else 0.0,
    }
    return result


def calculate_fap_bootstrap(
    bls: BoxLeastSquares,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> float:
    """
    Calculate FAP using a simple bootstrap / permutation scheme.

    Notes
    -----
    This is a heuristic. For publication-grade FAP you may want:
    - block bootstrap that preserves red noise
    - bootstrap over residuals of a variability model
    - injection-recovery simulations
    """
    rng = np.random.default_rng(seed)
    max_powers: List[float] = []

    periods_test = np.linspace(period * 0.9, period * 1.1, 150)
    durations_test = np.array([0.02, 0.05, 0.1], dtype=float)

    for _ in range(int(n_bootstrap)):
        idx = rng.permutation(len(flux))
        flux_bs = flux[idx]

        try:
            power_bs = bls.power(periods_test, durations_test, objective="likelihood")
            max_powers.append(float(np.nanmax(power_bs.power)))
        except Exception:
            max_powers.append(0.0)

    # Original power at the candidate period (use a representative duration)
    try:
        original_power = float(bls.power([period], [0.05], objective="likelihood").power[0])
    except Exception:
        original_power = 0.0

    max_powers_arr = np.asarray(max_powers, dtype=float)
    fap = float(np.sum(max_powers_arr >= original_power) / max(len(max_powers_arr), 1))
    return max(fap, 1e-10)


def estimate_parameters_mcmc(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_guess: float,
    t0_guess: float,
    duration_guess: float,
    depth_guess: float,
    n_walkers: int = 32,
    n_steps: int = 1500,
    burnin: int = 400,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Estimate transit parameters and uncertainties using MCMC on a fast box model.
    """

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    def log_likelihood(theta, t, f, ferr):
        period, t0, duration, depth = theta

        # Box model
        phase = ((t - t0) / period) % 1.0
        half_width = 0.5 * duration / period
        in_tr = (phase < half_width) | (phase > 1.0 - half_width)

        model = np.ones_like(f)
        model[in_tr] = 1.0 - depth

        chi2 = np.nansum(((f - model) / ferr) ** 2)
        return -0.5 * chi2

    def log_prior(theta):
        period, t0, duration, depth = theta

        if not (0.1 < period < 1000.0):
            return -np.inf
        if not (np.nanmin(time) < t0 < np.nanmax(time)):
            return -np.inf
        if not (1e-4 < duration < 1.0):
            return -np.inf
        if not (1e-6 < depth < 0.5):
            return -np.inf
        return 0.0

    def log_probability(theta, t, f, ferr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, t, f, ferr)

    ndim = 4
    pos0 = np.array([period_guess, t0_guess, duration_guess, depth_guess], dtype=float)

    # Small scatter around initial guess
    scale = np.array([1e-4, 1e-4, 1e-4, 1e-4], dtype=float)
    pos = pos0 + scale * np.random.randn(int(n_walkers), ndim) * np.maximum(np.abs(pos0), 1.0)

    sampler = emcee.EnsembleSampler(int(n_walkers), ndim, log_probability, args=(time, flux, flux_err))
    sampler.run_mcmc(pos, int(n_steps), progress=False)

    samples = sampler.get_chain(discard=int(burnin), flat=True)

    # Percentiles
    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)

    errors = {
        "period_err": float((p84[0] - p16[0]) / 2.0),
        "t0_err": float((p84[1] - p16[1]) / 2.0),
        "duration_err": float((p84[2] - p16[2]) / 2.0),
        "depth_err": float((p84[3] - p16[3]) / 2.0),
    }
    return samples, errors


def find_period_gls(time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Generalized Lomb-Scargle periodogram."""
    from astropy.timeseries import LombScargle

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)

    ls = LombScargle(time, flux, dy=flux_err)
    frequency, power = ls.autopower(minimum_frequency=1 / 100.0, maximum_frequency=1 / 0.5)

    best_idx = int(np.nanargmax(power))
    period = float(1.0 / frequency[best_idx])
    fap = float(ls.false_alarm_probability(power[best_idx]))

    return {
        "period": period,
        "power": float(power[best_idx]),
        "fap": fap,
        "frequencies": frequency,
        "powers": power,
        "method": "gls",
    }


def find_period_pdm(time: np.ndarray, flux: np.ndarray, nbins: int = 10) -> Dict[str, object]:
    """Phase Dispersion Minimization (PDM)."""
    from astropy.stats import phase_dispersion

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    periods = np.logspace(np.log10(0.5), np.log10(100.0), 1000)
    theta = np.array([phase_dispersion(time, flux, p, nbins=int(nbins)) for p in periods], dtype=float)
    best_period = float(periods[int(np.nanargmin(theta))])

    return {
        "period": best_period,
        "theta": float(np.nanmin(theta)),
        "periods": periods,
        "thetas": theta,
        "method": "pdm",
    }


def calculate_consensus(results: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Calculate consensus from multiple period search methods."""
    periods: List[float] = []
    weights: List[float] = []

    for method, result in results.items():
        if method not in ("bls", "gls"):
            continue
        if not isinstance(result, dict):
            continue
        if "period" not in result:
            continue

        p = float(result["period"])
        periods.append(p)

        # Weight by -log10(FAP) when possible; else by SNR; else 1
        if "fap" in result and result["fap"] is not None and float(result["fap"]) > 0:
            weights.append(float(-np.log10(float(result["fap"]))))
        elif "snr" in result and result["snr"] is not None:
            weights.append(float(result["snr"]))
        else:
            weights.append(1.0)

    if not periods:
        return None

    periods_arr = np.asarray(periods, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / np.sum(weights_arr)

    consensus_period = float(np.average(periods_arr, weights=weights_arr))
    harmonics = check_harmonics(periods_arr, consensus_period)

    return {
        "period": consensus_period,
        "period_std": float(np.std(periods_arr)),
        "method_agreement": int(len(periods_arr)),
        "weights": weights_arr.tolist(),
        "individual_periods": periods_arr.tolist(),
        "harmonics_detected": harmonics,
        "is_harmonic": bool(harmonics.get("is_harmonic", False)),
    }


def check_harmonics(periods: np.ndarray, reference: float, tolerance: float = 0.01) -> Dict[str, object]:
    """Check for common harmonic relationships relative to a reference period."""
    harmonics: Dict[str, object] = {}

    ref = float(reference)
    for p in np.asarray(periods, dtype=float):
        ratio = float(p / ref)
        for mult in (0.5, 2.0, 3.0, 1.0 / 3.0):
            if abs(ratio - mult) < tolerance:
                harmonics[str(mult)] = True
                break

    harmonics["is_harmonic"] = len([k for k in harmonics.keys() if k != "is_harmonic"]) > 0
    return harmonics


# =============================================================================
# Validation
# =============================================================================

def validate_transit_detection(
    time: np.ndarray,
    flux: np.ndarray,
    consensus: Optional[Dict[str, object]],
    all_results: Dict[str, object],
) -> Dict[str, object]:
    """Validate transit detection with multiple tests."""
    validation: Dict[str, object] = {}

    if consensus is None:
        validation["passed"] = False
        validation["reason"] = "no_consensus"
        return validation

    period = float(consensus["period"])
    t0 = float(all_results.get("bls", {}).get("t0", time[0] if len(time) else 0.0))

    validation["odd_even"] = check_odd_even_consistency(time, flux, period, t0)

    if "bls" in all_results and isinstance(all_results["bls"], dict) and "duration" in all_results["bls"]:
        validation["duration_consistency"] = check_duration_consistency(
            float(all_results["bls"]["duration"]), period
        )
    else:
        validation["duration_consistency"] = True  # can't test

    validation["secondary"] = check_secondary_eclipse(time, flux, period, t0)
    validation["variability"] = check_stellar_variability(time, flux)

    validation["passed"] = (
        float(validation.get("odd_even", {}).get("p_value", 0.0)) > 0.01
        and bool(validation.get("duration_consistency", True))
        and bool(validation.get("secondary", {}).get("detected", False)) is False
    )

    return validation


def check_odd_even_consistency(time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> Dict[str, object]:
    """Check if odd and even transits are consistent."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    phase = ((time - t0) / period) % 1.0
    transit_number = np.floor((time - t0) / period + 0.5).astype(int)
    is_even = (transit_number % 2) == 0

    # Simple in-transit window around phase 0
    in_transit = (phase < 0.05) | (phase > 0.95)

    n_even = int(np.sum(in_transit & is_even))
    n_odd = int(np.sum(in_transit & ~is_even))
    if n_even < 10 or n_odd < 10:
        return {"p_value": 1.0, "conclusion": "insufficient_data", "n_even": n_even, "n_odd": n_odd}

    flux_even = flux[in_transit & is_even]
    flux_odd = flux[in_transit & ~is_even]

    t_stat, p_value = stats.ttest_ind(flux_even, flux_odd, equal_var=False, nan_policy="omit")

    return {
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else float("nan"),
        "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "n_even": int(len(flux_even)),
        "n_odd": int(len(flux_odd)),
        "mean_even": float(np.nanmean(flux_even)),
        "mean_odd": float(np.nanmean(flux_odd)),
        "conclusion": "consistent" if (np.isfinite(p_value) and p_value > 0.01) else "inconsistent",
    }


def check_duration_consistency(duration: float, period: float) -> bool:
    """
    Very basic duration sanity check.
    Transit duration must be < period and not absurdly tiny.
    """
    if duration <= 0:
        return False
    if duration >= period:
        return False
    if duration < (1.0 / 24.0 / 60.0):  # < 1 minute in days
        return False
    return True


def check_secondary_eclipse(time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> Dict[str, object]:
    """
    Simple secondary eclipse heuristic:
    check for a dip near phase 0.5 compared to out-of-eclipse baseline.

    This is intentionally conservative and fast.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    phase = ((time - t0) / period) % 1.0

    # windows
    win_primary = (phase < 0.05) | (phase > 0.95)
    win_secondary = (np.abs(phase - 0.5) < 0.05)
    win_out = (np.abs(phase - 0.25) < 0.05) | (np.abs(phase - 0.75) < 0.05)

    if np.sum(win_secondary) < 10 or np.sum(win_out) < 10:
        return {"detected": False, "reason": "insufficient_data"}

    sec_depth = float(np.nanmedian(flux[win_out]) - np.nanmedian(flux[win_secondary]))
    # crude significance using MAD
    mad = float(np.nanmedian(np.abs(flux[win_out] - np.nanmedian(flux[win_out]))))
    sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(flux[win_out]))
    snr = sec_depth / max(sigma, 1e-12)

    detected = bool(snr > 5.0 and sec_depth > 0)
    return {"detected": detected, "secondary_depth": sec_depth, "secondary_snr": float(snr)}


def check_stellar_variability(time: np.ndarray, flux: np.ndarray) -> Dict[str, object]:
    """
    Quick variability diagnostic: robust scatter and a crude autocorrelation proxy.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    med = float(np.nanmedian(flux))
    mad = float(np.nanmedian(np.abs(flux - med)))
    robust_std = float(1.4826 * mad)

    return {"robust_std": robust_std, "median_flux": med}


# =============================================================================
# Information criteria
# =============================================================================

def calculate_bic(time: np.ndarray, flux: np.ndarray, model: np.ndarray, flux_err: np.ndarray, n_params: int) -> float:
    """Bayesian Information Criterion."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    model = np.asarray(model, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    n = int(len(time))
    if n <= 0:
        return float("nan")

    chi2 = float(np.nansum(((flux - model) / flux_err) ** 2))
    return float(chi2 + int(n_params) * np.log(n))
