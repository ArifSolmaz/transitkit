"""Core transit analysis functions"""

import numpy as np


def generate_transit_signal(time, period=10.0, depth=0.01, duration=0.1):
    """
    Generate a synthetic transit signal.

    Parameters
    ----------
    time : array
        Time array in days
    period : float
        Orbital period in days
    depth : float
        Transit depth (fractional)
    duration : float
        Transit duration in days

    Returns
    -------
    flux : array
        Flux with transit signal
    """
    time = np.asarray(time, dtype=float)
    flux = np.ones_like(time)

    # Add transits at each period
    t0 = period / 2  # First transit at time = period/2
    n_transits = int(time[-1] / period) + 2

    for i in range(n_transits):
        transit_time = t0 + i * period
        in_transit = (time > transit_time - duration / 2) & (time < transit_time + duration / 2)
        flux[in_transit] = 1 - depth

    return flux


def add_noise(flux, noise_level=0.001):
    """
    Add Gaussian noise to flux.

    Parameters
    ----------
    flux : array
        Input flux array
    noise_level : float
        Standard deviation of noise

    Returns
    -------
    noisy_flux : array
        Flux with added noise
    """
    flux = np.asarray(flux, dtype=float)
    noise = np.random.normal(0, noise_level, len(flux))
    return flux + noise


def plot_light_curve(time, flux, title="Light Curve"):
    """
    Plot a light curve.

    Parameters
    ----------
    time : array
        Time array
    flux : array
        Flux array
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(time, flux, "k.", alpha=0.5, markersize=1)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def find_transits_box(
    time,
    flux,
    min_period=0.5,
    max_period=100.0,
    durations=None,
    n_periods=5000,
):
    """
    Box Least Squares period search using astropy (robust, CI-friendly).

    Returns
    -------
    results : dict
        Includes:
          - period, t0, duration, depth
          - all_periods, all_power
        Backward compatible aliases:
          - score (best power)
          - all_scores (same as all_power)
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Duration grid (days)
    if durations is None:
        # Keep durations within a sensible range. You can tune these later.
        durations = np.linspace(0.02, 0.3, 20)

    periods = np.linspace(min_period, max_period, int(n_periods))

    bls = BoxLeastSquares(time, flux)
    power = bls.power(periods, durations)

    k = int(np.nanargmax(power.power))
    best_power = float(power.power[k])

    return {
        # Best-fit solution
        "period": float(power.period[k]),
        "t0": float(power.transit_time[k]),
        "duration": float(power.duration[k]),
        "depth": float(power.depth[k]),
        "snr": float(power.snr[k]) if hasattr(power, "snr") else None,

        # Full scan
        "all_periods": power.period,
        "all_power": power.power,

        # Backward-compatible keys (fixes your CI test expectations)
        "score": best_power,
        "all_scores": power.power,
    }

