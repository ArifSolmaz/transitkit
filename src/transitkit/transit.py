"""Core transit analysis functions"""

import numpy as np
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 4))
    plt.plot(time, flux, 'k.', alpha=0.5, markersize=1)
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Flux')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def find_transits_box(time, flux, min_period=0.5, max_period=100.0):
    """
    Simple box transit search (simplified BLS).
    
    Parameters
    ----------
    time : array
        Time array
    flux : array
        Flux array
    min_period : float
        Minimum period to search
    max_period : float
        Maximum period to search
    
    Returns
    -------
    results : dict
        Dictionary with best period and statistics
    """
    # Simplified version - in real code, implement BLS algorithm
    periods = np.linspace(min_period, max_period, 100)
    scores = []
    
    for period in periods:
        # Phase fold
        phase = (time / period) % 1
        
        # Simple box search
        in_transit = (phase < 0.1) | (phase > 0.9)
        out_transit = ~in_transit
        
        if np.sum(in_transit) > 0 and np.sum(out_transit) > 0:
            depth = np.mean(flux[out_transit]) - np.mean(flux[in_transit])
            score = depth * np.sqrt(np.sum(in_transit))  # Simple SNR-like score
            scores.append(score)
        else:
            scores.append(0)
    
    best_idx = np.argmax(scores)
    
    return {
        'period': periods[best_idx],
        'score': scores[best_idx],
        'all_periods': periods,
        'all_scores': scores
    }