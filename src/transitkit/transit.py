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
    flux = np.ones_like(time)

    # Add transits at each period
    t0 = period / 2  # First transit at time = period/2
    n_transits = int(time[-1] / period) + 2

    for i in range(n_transits):
        transit_time = t0 + i * period
        in_transit = (time > transit_time - duration/2) & (time < transit_time + duration/2)
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

def find_transits_box(time, flux, min_period=0.5, max_period=100.0,
                      durations=None, n_periods=5000):
    """
    Box Least Squares period search using astropy (robust, CI-friendly).
    Returns best period, t0, duration, depth, and full power arrays.
    """
    import numpy as np
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if durations is None:
        # Reasonable default duration grid (days)
        baseline = np.nanmax(time) - np.nanmin(time)
        # Keep durations in a sensible range
        durations = np.linspace(0.02, 0.3, 20)

    periods = np.linspace(min_period, max_period, n_periods)

    bls = BoxLeastSquares(time, flux)
    power = bls.power(periods, durations)

    k = int(np.nanargmax(power.power))

    return {
        "period": float(power.period[k]),
        "t0": float(power.transit_time[k]),
        "duration": float(power.duration[k]),
        "depth": float(power.depth[k]),
        "snr": float(power.snr[k]) if hasattr(power, "snr") else None,
        "all_periods": power.period,
        "all_power": power.power,
    }

