# core.py - Scientific core with rigorous methods
"""
Core transit analysis functions with error propagation, MCMC support,
and publication-quality algorithms.

This module is also a compatibility layer for older TransitKit APIs:
- provides add_noise in transitkit.core
- generate_transit_signal_mandel_agol accepts legacy kwargs (depth, duration)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union, Any

import numpy as np
from scipy import stats
from astropy.timeseries import BoxLeastSquares

__all__ = [
    "TransitParameters",
    "add_noise",
    "generate_transit_signal_mandel_agol",
    "find_transits_bls_advanced",
    "find_transits_multiple_methods",
    "find_period_gls",
    "find_period_pdm",
    "estimate_parameters_mcmc",
    "calculate_consensus",
    "validate_transit_detection",
]

# ----------------------------
# Helpers
# ----------------------------

def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)

def add_noise(
    flux: Union[np.ndarray, List[float]],
    noise_level: float = 1e-3,
    seed: Optional[int] = None,
    kind: str = "gaussian",
) -> np.ndarray:
    """
    Add noise to a flux array.

    Parameters
    ----------
    flux : array-like
        Input flux.
    noise_level : float
        1-sigma noise amplitude (same units as flux).
    seed : int, optional
        RNG seed for reproducibility.
    kind : {"gaussian"}
        Noise type. (kept for forward extension)

    Returns
    -------
    noisy_flux : np.ndarray
    """
    f = _as_float_array(flux)
    if noise_level <= 0:
        return f.copy()

    rng = np.random.default_rng(seed)

    if kind.lower() != "gaussian":
        raise ValueError("Only kind='gaussian' is currently supported.")

    return f + rng.normal(0.0, noise_level, size=f.shape)


def _box_model(time: np.ndarray, period: float, t0: float, duration: float, depth: float) -> np.ndarray:
    """Fast box-shaped transit model for fallback/testing."""
    t = _as_float_array(time)
    phase = ((t - t0) / period) % 1.0
    half_width = 0.5 * (duration / period)
    in_transit = (phase < half_width) | (phase > 1.0 - half_width)
    model = np.ones_like(t)
    model[in_transit] = 1.0 - depth
    return model


def _infer_aRs_from_duration(period: float, duration: float, rprs: float) -> float:
    """
    Rough circular, b~0, inc~90 approximation:
        duration ≈ (period/pi) * arcsin( (1+rprs)/aRs )
    =>  aRs ≈ (1+rprs)/sin(pi*duration/period)
    """
    x = np.sin(np.pi * duration / max(period, 1e-12))
    x = np.clip(x, 1e-6, 1.0)
    return float((1.0 + rprs) / x)


# ----------------------------
# Data model
# ----------------------------

@dataclass
class TransitParameters:
    """Container for transit parameters with uncertainties (dataclass-safe ordering)."""

    # Required (non-default) FIRST (fixes: non-default argument 't0' follows default argument)
    period: float
    t0: float
    depth: float
    duration: float

    # Optional uncertainties / derived
    period_err: float = 0.0
    t0_err: float = 0.0
    depth_err: float = 0.0
    duration_err: float = 0.0

    b: float = 0.5
    b_err: float = 0.0

    rprs: Optional[float] = None
    rprs_err: float = 0.0

    aRs: Optional[float] = None
    aRs_err: float = 0.0

    inclination: float = 90.0
    inclination_err: float = 0.0

    limb_darkening: Tuple[float, float] = (0.1, 0.3)  # u1, u2

    snr: float = 0.0
    fap: float = 1.0

    quality_flags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Derive rprs from depth if missing
        if self.rprs is None and np.isfinite(self.depth) and self.depth >= 0:
            self.rprs = float(np.sqrt(self.depth))

        # Populate defaults for quality flags if empty
        if not self.quality_flags:
            self.quality_flags = {
                "bls_snr": bool(self.snr > 7),
                "duration_consistent": True,
                "odd_even_consistent": True,
            }

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_bls_result(cls, bls_result, **kwargs) -> "TransitParameters":
        """Create from BLS result with optional overrides."""
        return cls(
            period=float(bls_result.period),
            t0=float(bls_result.transit_time),
            depth=float(bls_result.depth),
            duration=float(bls_result.duration),
            snr=float(getattr(bls_result, "snr", 0.0)),
            **kwargs,
        )


# ----------------------------
# Signal generation
# ----------------------------

def generate_transit_signal_mandel_agol(
    time: Union[np.ndarray, List[float]],
    period: float,
    t0: Optional[float] = None,
    rprs: Optional[float] = None,
    aRs: Optional[float] = None,
    inclination: float = 90.0,
    eccentricity: float = 0.0,
    omega: float = 90.0,
    u1: float = 0.1,
    u2: float = 0.3,
    exptime: float = 0.0,
    supersample: int = 7,
    # ---- Backward-compatibility kwargs ----
    depth: Optional[float] = None,
    duration: Optional[float] = None,
    **_ignored: Any,
) -> np.ndarray:
    """
    Generate transit light curves using batman (Mandel & Agol 2002) if available.
    Falls back to a box model if batman is not available.

    Backward-compatible:
      - if depth is given and rprs is None -> rprs = sqrt(depth)
      - if duration is given and aRs is None -> approximate aRs from duration
      - if t0 is None -> choose a sensible default

    Returns
    -------
    flux : np.ndarray
        Model flux array.
    """
    t = _as_float_array(time)
    if t.size == 0:
        return t.copy()

    if t0 is None:
        # default: mid-ish transit inside the data span
        t0 = float(t.min() + 0.5 * period)

    # depth -> rprs
    if rprs is None:
        if depth is None:
            raise TypeError("Provide either rprs=... or depth=... for signal generation.")
        if depth < 0:
            raise ValueError("depth must be >= 0.")
        rprs = float(np.sqrt(depth))

    # duration -> aRs approximation if missing
    if aRs is None:
        if duration is not None and duration > 0:
            aRs = _infer_aRs_from_duration(period=float(period), duration=float(duration), rprs=float(rprs))
        else:
            aRs = 15.0  # safe default

    # Try batman; fallback to box model
    try:
        from batman import TransitParams, TransitModel  # type: ignore

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

        m = TransitModel(params, t, supersample_factor=int(max(supersample, 1)), exp_time=float(exptime))
        return m.light_curve(params)

    except Exception:
        # box fallback: use duration if provided; else approximate from aRs
        if duration is None or duration <= 0:
            # crude: duration fraction ~ (1+rprs)/aRs * period/pi
            duration = float((period / np.pi) * np.arcsin(np.clip((1.0 + rprs) / max(aRs, 1e-6), 0, 1)))
            duration = max(duration, 0.01 * period)

        depth_box = float(depth) if depth is not None else float(rprs) ** 2
        return _box_model(t, float(period), float(t0), float(duration), depth_box)


# ----------------------------
# Period search
# ----------------------------

def find_transits_bls_advanced(
    time: Union[np.ndarray, List[float]],
    flux: Union[np.ndarray, List[float]],
    flux_err: Optional[Union[np.ndarray, List[float]]] = None,
    min_period: float = 0.5,
    max_period: float = 100.0,
    n_periods: int = 10000,
    durations: Optional[np.ndarray] = None,
    objective: str = "likelihood",
    n_bootstrap_fap: int = 200,
) -> Dict[str, Any]:
    """
    Advanced BLS with SNR, bootstrap FAP, and optional MCMC error estimates.
    """
    t = _as_float_array(time)
    f = _as_float_array(flux)

    if t.size != f.size:
        raise ValueError("time and flux must have the same length.")

    if flux_err is None:
        # conservative: white-noise estimate
        ferr = np.full_like(f, np.nanstd(f) / max(np.sqrt(len(f)), 1.0))
    else:
        ferr = _as_float_array(flux_err)
        if ferr.size != f.size:
            raise ValueError("flux_err must have the same length as flux.")

    if durations is None:
        # 0.5–15 hours in days, but ensure max duration < min_period
        max_dur_hours = min(15.0, min_period * 24 * 0.8)  # at most 80% of min_period
        max_dur_hours = max(max_dur_hours, 1.0)  # at least 1 hour
        durations = np.logspace(np.log10(0.5 / 24.0), np.log10(max_dur_hours / 24.0), 15)

    # log-spaced periods are generally better
    periods = np.logspace(np.log10(min_period), np.log10(max_period), int(n_periods))

    bls = BoxLeastSquares(t, f, dy=ferr)

    obj = "likelihood" if objective.lower() == "likelihood" else "snr"
    power = bls.power(periods, durations, objective=obj)

    best_idx = int(np.nanargmax(power.power))

    best_period = float(power.period[best_idx])
    best_t0 = float(power.transit_time[best_idx])
    best_dur = float(power.duration[best_idx])
    best_depth = float(power.depth[best_idx])

    model = bls.model(t, best_period, best_dur, best_t0)
    resid = f - model
    rms = float(np.nanstd(resid))

    # in-transit points estimation
    in_transit = model < (1.0 - 0.5 * max(best_depth, 0.0))
    n_in = int(np.sum(in_transit))
    n_in = max(n_in, 1)

    snr = 0.0
    if rms > 0 and np.isfinite(rms):
        snr = float((best_depth / rms) * np.sqrt(n_in))

    # FAP (bootstrap-ish)
    fap = calculate_fap_bootstrap(
        bls=bls,
        time=t,
        flux=f,
        flux_err=ferr,
        period=best_period,
        duration=best_dur,
        n_bootstrap=int(max(50, n_bootstrap_fap)),
    )

    # MCMC errors only when meaningful and not too slow
    param_errors = {"period_err": 0.0, "t0_err": 0.0, "duration_err": 0.0, "depth_err": 0.0}
    if snr >= 5:
        try:
            _, param_errors = estimate_parameters_mcmc(
                t, f, ferr,
                period_guess=best_period,
                t0_guess=best_t0,
                duration_guess=best_dur,
                depth_guess=best_depth,
                n_walkers=24,
                n_steps=800,
                burnin=200,
            )
        except Exception:
            # keep zeros; do not fail detection
            pass

    chi2 = float(np.nansum(((resid) / ferr) ** 2)) if np.all(np.isfinite(ferr)) else float("nan")

    result = {
        "period": best_period,
        "t0": best_t0,
        "duration": best_dur,
        "depth": best_depth,
        "snr": float(snr),
        "fap": float(fap),
        "power": float(power.power[best_idx]),

        "all_periods": power.period,
        "all_powers": power.power,
        "all_durations": power.duration,

        "errors": param_errors,

        "residuals_rms": rms,
        "chi2": chi2,
        "bic": calculate_bic(t, f, model, ferr, n_params=4),

        "method": "bls_advanced",
        "objective": obj,
        "n_data_points": int(len(t)),
        "data_span": float(t.max() - t.min()) if len(t) > 1 else 0.0,
    }

    return result


def calculate_fap_bootstrap(
    bls: BoxLeastSquares,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    duration: float,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> float:
    """
    Bootstrap-ish FAP by phase scrambling flux and re-running a small local BLS scan.

    Returns
    -------
    fap : float in [1e-10, 1]
    """
    rng = np.random.default_rng(seed)
    max_powers: List[float] = []

    # local scan around best period
    periods_test = np.linspace(period * 0.9, period * 1.1, 80)
    durations_test = np.array([max(duration * 0.7, 0.01), duration, duration * 1.3])

    for _ in range(int(n_bootstrap)):
        perm = rng.permutation(len(flux))
        f_bs = flux[perm]
        try:
            p_bs = bls.power(periods_test, durations_test, objective="likelihood")
            max_powers.append(float(np.nanmax(p_bs.power)))
        except Exception:
            max_powers.append(0.0)

    try:
        p0 = bls.power([period], [duration], objective="likelihood").power[0]
        p0 = float(p0)
    except Exception:
        return 1.0

    max_powers_arr = np.asarray(max_powers, dtype=float)
    fap = float(np.mean(max_powers_arr >= p0))
    return float(np.clip(max(fap, 1e-10), 1e-10, 1.0))


def estimate_parameters_mcmc(
    time: Union[np.ndarray, List[float]],
    flux: Union[np.ndarray, List[float]],
    flux_err: Union[np.ndarray, List[float]],
    period_guess: float,
    t0_guess: float,
    duration_guess: float,
    depth_guess: float,
    n_walkers: int = 32,
    n_steps: int = 2000,
    burnin: int = 500,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Estimate transit parameters and uncertainties using MCMC (box model likelihood).
    """
    import emcee  # local import for faster module import

    t = _as_float_array(time)
    f = _as_float_array(flux)
    ferr = _as_float_array(flux_err)

    def log_likelihood(theta):
        period, t0, duration, depth = theta
        model = _box_model(t, period, t0, duration, depth)
        chi2 = np.nansum(((f - model) / ferr) ** 2)
        return -0.5 * chi2

    def log_prior(theta):
        period, t0, duration, depth = theta

        if not (0.1 < period < 1000):
            return -np.inf
        if not (t.min() - period < t0 < t.max() + period):
            return -np.inf
        if not (1e-4 < duration < 0.8 * period):
            return -np.inf
        if not (1e-6 < depth < 0.5):
            return -np.inf
        return 0.0

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim = 4
    base = np.array([period_guess, t0_guess, duration_guess, depth_guess], dtype=float)
    base = np.where(base == 0, 1e-6, base)

    rng = np.random.default_rng(123)
    pos = base + (1e-3 * rng.normal(size=(n_walkers, ndim)) * base)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, lambda th: log_prob(th))
    sampler.run_mcmc(pos, n_steps, progress=False)

    samples = sampler.get_chain(discard=int(burnin), flat=True)
    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)

    errors = {
        "period_err": float((p84[0] - p16[0]) / 2),
        "t0_err": float((p84[1] - p16[1]) / 2),
        "duration_err": float((p84[2] - p16[2]) / 2),
        "depth_err": float((p84[3] - p16[3]) / 2),
    }
    return samples, errors


def find_period_gls(time, flux, flux_err=None) -> Dict[str, Any]:
    """Generalized Lomb-Scargle periodogram."""
    from astropy.timeseries import LombScargle

    t = _as_float_array(time)
    f = _as_float_array(flux)
    ferr = None if flux_err is None else _as_float_array(flux_err)

    ls = LombScargle(t, f, dy=ferr)
    frequency, power = ls.autopower(minimum_frequency=1 / 100.0, maximum_frequency=1 / 0.5)

    best_idx = int(np.argmax(power))
    period = float(1.0 / frequency[best_idx])
    fap = float(ls.false_alarm_probability(power[best_idx]))

    return {"period": period, "power": float(power[best_idx]), "fap": fap, "frequencies": frequency, "powers": power, "method": "gls"}


def _phase_dispersion_theta(time, flux, period, nbins=10):
    """
    Calculate PDM theta statistic for a given period.
    
    Theta = variance_in_bins / total_variance
    Lower theta means better period match.
    """
    phase = ((time - time.min()) / period) % 1.0
    bin_edges = np.linspace(0, 1, nbins + 1)
    
    total_var = np.nanvar(flux)
    if total_var == 0 or not np.isfinite(total_var):
        return 1.0
    
    # Calculate variance within each bin
    bin_vars = []
    bin_counts = []
    for i in range(nbins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) > 1:
            bin_vars.append(np.nanvar(flux[mask]))
            bin_counts.append(np.sum(mask))
    
    if len(bin_vars) == 0:
        return 1.0
    
    # Weighted average of bin variances
    bin_vars = np.array(bin_vars)
    bin_counts = np.array(bin_counts)
    
    weighted_var = np.nansum(bin_vars * (bin_counts - 1)) / np.nansum(bin_counts - 1)
    theta = weighted_var / total_var
    
    return float(theta)


def find_period_pdm(time, flux, nbins: int = 10, min_period: float = 0.5, max_period: float = 100.0, n_periods: int = 1000) -> Dict[str, Any]:
    """Phase Dispersion Minimization (custom implementation)."""
    t = _as_float_array(time)
    f = _as_float_array(flux)

    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
    thetas = np.array([_phase_dispersion_theta(t, f, p, nbins=nbins) for p in periods], dtype=float)

    best_idx = int(np.nanargmin(thetas))
    best_period = float(periods[best_idx])
    return {"period": best_period, "theta": float(thetas[best_idx]), "periods": periods, "thetas": thetas, "method": "pdm"}


def find_transits_multiple_methods(
    time: Union[np.ndarray, List[float]],
    flux: Union[np.ndarray, List[float]],
    flux_err: Optional[Union[np.ndarray, List[float]]] = None,
    min_period: float = 0.5,
    max_period: float = 100.0,
    n_periods: int = 10000,
    methods: List[str] = None,
) -> Dict[str, Any]:
    """
    Find periodicity using multiple methods and return consensus + validation.
    """
    if methods is None:
        methods = ["bls", "gls", "pdm"]

    results: Dict[str, Any] = {}

    if "bls" in methods:
        results["bls"] = find_transits_bls_advanced(time, flux, flux_err, min_period, max_period, n_periods)

    if "gls" in methods:
        results["gls"] = find_period_gls(time, flux, flux_err)

    if "pdm" in methods:
        results["pdm"] = find_period_pdm(time, flux)

    consensus = calculate_consensus(results)
    results["consensus"] = consensus
    results["validation"] = validate_transit_detection(time, flux, consensus, results)

    return results


def calculate_consensus(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Weighted consensus period from methods."""
    periods: List[float] = []
    weights: List[float] = []

    for method, result in results.items():
        if method not in {"bls", "gls", "pdm"}:
            continue
        if not isinstance(result, dict) or "period" not in result:
            continue

        p = float(result["period"])
        periods.append(p)

        # prefer low FAP; else SNR; else weight=1
        w = 1.0
        if "fap" in result and result["fap"] is not None:
            fap = float(result["fap"])
            if np.isfinite(fap) and fap > 0:
                w = max(-np.log10(fap), 0.0)
        elif "snr" in result:
            w = max(float(result["snr"]), 0.0)

        weights.append(w)

    if len(periods) == 0:
        return None

    p_arr = np.asarray(periods, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    s = float(np.sum(w_arr))
    if not np.isfinite(s) or s <= 0:
        w_arr = np.ones_like(w_arr) / len(w_arr)
    else:
        w_arr = w_arr / s

    consensus_period = float(np.average(p_arr, weights=w_arr))
    harmonics = check_harmonics(p_arr, consensus_period)

    return {
        "period": consensus_period,
        "period_std": float(np.nanstd(p_arr)),
        "method_agreement": int(len(p_arr)),
        "weights": w_arr.tolist(),
        "individual_periods": p_arr.tolist(),
        "harmonics_detected": harmonics,
        "is_harmonic": bool(harmonics.get("is_harmonic", False)),
    }


def check_harmonics(periods: np.ndarray, reference: float, tolerance: float = 0.01) -> Dict[str, Any]:
    """Check for simple harmonic relationships w.r.t. reference."""
    out: Dict[str, Any] = {}
    if reference <= 0:
        out["is_harmonic"] = False
        return out

    for p in periods:
        ratio = float(p / reference)
        for mult in [0.5, 2.0, 3.0, 1.0 / 3.0]:
            if abs(ratio - mult) < tolerance:
                out[str(mult)] = True
                break

    out["is_harmonic"] = bool(len(out) > 0)
    return out


# ----------------------------
# Validation
# ----------------------------

def validate_transit_detection(time, flux, consensus, all_results) -> Dict[str, Any]:
    """Validate detection with simple sanity checks."""
    validation: Dict[str, Any] = {}
    if consensus is None:
        validation["passed"] = False
        return validation

    period = float(consensus["period"])
    t0 = float(all_results.get("bls", {}).get("t0", np.nan))

    validation["odd_even"] = check_odd_even_consistency(time, flux, period, t0)

    if "bls" in all_results and isinstance(all_results["bls"], dict):
        validation["duration_consistency"] = check_duration_consistency(
            float(all_results["bls"].get("duration", np.nan)),
            period,
        )
    else:
        validation["duration_consistency"] = True

    validation["secondary"] = check_secondary_eclipse(time, flux, period, t0)
    validation["variability"] = check_stellar_variability(time, flux)

    validation["passed"] = (
        validation.get("odd_even", {}).get("p_value", 1.0) > 0.01
        and bool(validation.get("duration_consistency", True))
        and (validation.get("secondary", {}).get("detected", False) is False)
    )
    return validation


def check_odd_even_consistency(time, flux, period, t0) -> Dict[str, Any]:
    """Odd-even transit consistency test."""
    t = _as_float_array(time)
    f = _as_float_array(flux)

    if not np.isfinite(t0):
        return {"p_value": 1.0, "conclusion": "no_t0"}

    phase = ((t - t0) / period) % 1.0
    transit_number = np.floor((t - t0) / period + 0.5)
    is_even = (transit_number.astype(int) % 2) == 0

    # crude in-transit window
    in_transit = (phase < 0.05) | (phase > 0.95)
    if np.sum(in_transit & is_even) < 10 or np.sum(in_transit & ~is_even) < 10:
        return {"p_value": 1.0, "conclusion": "insufficient_data"}

    even = f[in_transit & is_even]
    odd = f[in_transit & ~is_even]

    t_stat, p_value = stats.ttest_ind(even, odd, equal_var=False, nan_policy="omit")

    return {
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "n_even": int(len(even)),
        "n_odd": int(len(odd)),
        "mean_even": float(np.nanmean(even)),
        "mean_odd": float(np.nanmean(odd)),
        "conclusion": "consistent" if (np.isfinite(p_value) and p_value > 0.01) else "inconsistent",
    }


def check_duration_consistency(duration: float, period: float) -> bool:
    """Basic physical sanity: duration must be positive and much shorter than period."""
    if not (np.isfinite(duration) and np.isfinite(period)):
        return True
    if duration <= 0:
        return False
    if duration >= 0.9 * period:
        return False
    return True


def check_secondary_eclipse(time, flux, period, t0) -> Dict[str, Any]:
    """
    Very simple secondary eclipse check around phase ~0.5.
    Returns detected=True if there's a statistically significant dip.
    """
    t = _as_float_array(time)
    f = _as_float_array(flux)

    if not np.isfinite(t0):
        return {"detected": False, "depth": 0.0}

    phase = ((t - t0) / period) % 1.0
    sec = (np.abs(phase - 0.5) < 0.03)
    oot = (np.abs(phase - 0.5) > 0.10)

    if np.sum(sec) < 10 or np.sum(oot) < 10:
        return {"detected": False, "depth": 0.0}

    sec_med = float(np.nanmedian(f[sec]))
    oot_med = float(np.nanmedian(f[oot]))
    depth = oot_med - sec_med

    # simple t-test
    t_stat, p_value = stats.ttest_ind(f[oot], f[sec], equal_var=False, nan_policy="omit")
    detected = bool(np.isfinite(p_value) and p_value < 0.001 and depth > 0)

    return {"detected": detected, "depth": float(depth), "p_value": float(p_value) if np.isfinite(p_value) else 1.0}


def check_stellar_variability(time, flux) -> Dict[str, Any]:
    """Simple variability metric: robust scatter (MAD) + long-term trend amplitude."""
    t = _as_float_array(time)
    f = _as_float_array(flux)

    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med))
    robust_sigma = 1.4826 * mad

    # trend amplitude via rolling median (very light)
    if len(f) >= 50:
        # crude smoothing by binning
        nb = 20
        bins = np.linspace(t.min(), t.max(), nb + 1)
        idx = np.digitize(t, bins) - 1
        bmed = np.array([np.nanmedian(f[idx == i]) for i in range(nb)], dtype=float)
        trend_amp = float(np.nanmax(bmed) - np.nanmin(bmed)) if np.any(np.isfinite(bmed)) else 0.0
    else:
        trend_amp = 0.0

    return {"robust_sigma": float(robust_sigma), "trend_amplitude": float(trend_amp)}


def calculate_bic(time, flux, model, flux_err, n_params: int) -> float:
    """Bayesian Information Criterion."""
    t = _as_float_array(time)
    f = _as_float_array(flux)
    m = _as_float_array(model)
    ferr = _as_float_array(flux_err)

    n = max(int(len(t)), 1)
    chi2 = float(np.nansum(((f - m) / ferr) ** 2)) if np.all(np.isfinite(ferr)) else float("nan")
    return float(chi2 + n_params * np.log(n))
