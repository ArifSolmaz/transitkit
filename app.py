"""
🌟 TransitKit - Professional Exoplanet Transit Analysis
A comprehensive Streamlit application for transit light curve analysis
Based on github.com/arifsolmaz/transitkit
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TransitKit - Exoplanet Transit Analysis",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/arifsolmaz/transitkit',
        'Report a bug': 'https://github.com/arifsolmaz/transitkit/issues',
        'About': "# TransitKit v2.0\nProfessional Exoplanet Transit Light Curve Analysis"
    }
)

# ============================================================================
# CUSTOM CSS - DARK SPACE THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #00d4aa;
        --secondary: #7c3aed;
        --accent: #f59e0b;
        --bg-dark: #0a0a0f;
        --bg-card: #12121a;
        --bg-elevated: #1a1a2e;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(0, 212, 170, 0.2);
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #12121a 0%, #1a1a2e 50%, #12121a 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent);
    }
    
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #7c3aed 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    
    .main-header .subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header .badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        color: var(--primary);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.75rem;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--primary);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.15);
    }
    
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .metric-card .label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .metric-card .unit {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-left: 0.25rem;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    
    .section-header h2 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    
    .section-header .icon {
        font-size: 1.3rem;
    }
    
    /* Info/Warning boxes */
    .info-box {
        background: rgba(0, 212, 170, 0.08);
        border-left: 3px solid var(--primary);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: var(--text-secondary);
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid var(--accent);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.08);
        border-left: 3px solid #22c55e;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Method comparison cards */
    .method-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .method-card.best {
        border-color: var(--primary);
        background: rgba(0, 212, 170, 0.05);
    }
    
    .method-name {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1rem;
    }
    
    /* Parameter display */
    .param-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .param-name {
        color: var(--text-secondary);
    }
    
    .param-value {
        font-family: 'JetBrains Mono', monospace;
        color: var(--primary);
    }
    
    /* Code blocks */
    .code-block {
        background: var(--bg-dark);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: var(--bg-dark) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: var(--primary) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3);
    }
    
    /* DataFrames */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
    
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress indicator */
    .transit-phase {
        display: flex;
        justify-content: space-between;
        background: var(--bg-card);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .phase-step {
        text-align: center;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    
    .phase-step.active {
        background: var(--primary);
        color: var(--bg-dark);
    }
    
    .phase-step.completed {
        background: rgba(0, 212, 170, 0.2);
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TRANSIT MODEL FUNCTIONS (Core TransitKit functionality)
# ============================================================================

def mandel_agol_transit(t, t0, period, rp_rs, a_rs, inc, u1=0.4, u2=0.2):
    """
    Compute Mandel & Agol (2002) quadratic limb-darkened transit model.
    Simplified implementation for demonstration.
    """
    # Phase fold
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    
    # Impact parameter
    b = a_rs * np.cos(np.radians(inc))
    
    # Approximate transit duration
    duration = period / np.pi * np.arcsin(
        np.sqrt((1 + rp_rs)**2 - b**2) / (a_rs * np.sin(np.radians(inc)))
    )
    
    # Normalized time from mid-transit
    x = np.abs(phase * period / (duration / 2))
    
    # Simple transit shape (trapezoidal approximation with limb darkening)
    flux = np.ones_like(t)
    
    in_transit = x < 1.0
    ingress_egress = (x >= 0.8) & (x < 1.0)
    
    # Limb darkening effect
    mu = np.sqrt(1 - np.minimum(x[in_transit], 1.0)**2)
    ld_factor = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2
    
    # Apply transit depth with limb darkening
    depth = rp_rs**2
    flux[in_transit] = 1 - depth * ld_factor
    
    # Smooth ingress/egress
    flux[ingress_egress] = 1 - depth * (1 - x[ingress_egress]) / 0.2 * ld_factor[x[in_transit] >= 0.8][:sum(ingress_egress)]
    
    return flux


def generate_transit_signal(time, period=5.0, t0=2.5, depth=0.01, duration=0.15,
                           u1=0.4, u2=0.2, inc=89.0, a_rs=15.0):
    """
    Generate a realistic transit light curve using Mandel-Agol model.
    """
    # Calculate rp_rs from depth
    rp_rs = np.sqrt(depth)
    
    # Generate flux
    flux = np.ones_like(time)
    
    # Find all transit events
    n_transits = int((time[-1] - time[0]) / period) + 2
    
    for i in range(-1, n_transits + 1):
        tc = t0 + i * period
        
        # Distance from transit center
        dt = np.abs(time - tc)
        
        # In-transit mask
        in_transit = dt < duration / 2
        
        if np.any(in_transit):
            # Normalized position (-1 to 1 during transit)
            x = (time[in_transit] - tc) / (duration / 2)
            
            # Impact parameter effect
            b = a_rs * np.cos(np.radians(inc))
            
            # Limb darkening
            z = np.sqrt(x**2 + (b / (1 + rp_rs))**2)
            z = np.clip(z, 0, 1)
            
            mu = np.sqrt(1 - z**2)
            ld_factor = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2
            
            # Transit depth with limb darkening
            flux[in_transit] = 1 - depth * ld_factor * (1 - z**2)
    
    return flux


def add_noise(flux, noise_level=0.001, stellar_var=0.0):
    """Add Gaussian noise and optional stellar variability."""
    noise = np.random.normal(0, noise_level, len(flux))
    
    if stellar_var > 0:
        # Add slow stellar variability
        t = np.arange(len(flux))
        variability = stellar_var * np.sin(2 * np.pi * t / len(flux) * 3)
        variability += stellar_var * 0.5 * np.sin(2 * np.pi * t / len(flux) * 7 + 1.5)
        flux = flux + variability
    
    return flux + noise


def box_least_squares(time, flux, min_period=1.0, max_period=20.0, 
                      n_periods=500, n_durations=5):
    """
    Optimized Box Least Squares (BLS) transit detection algorithm.
    """
    periods = np.linspace(min_period, max_period, n_periods)
    power = np.zeros(len(periods))
    best_params = {'period': 0, 'depth': 0, 't0': 0, 'duration': 0, 'snr': 0}
    max_power = 0
    
    flux_norm = flux - np.median(flux)
    n_bins = 50  # Fixed bin count for speed
    
    # Pre-compute duration grid
    duration_fracs = np.linspace(0.01, 0.15, n_durations)
    
    for i, period in enumerate(periods):
        phase = (time % period) / period
        
        # Bin the data using histogram (vectorized)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        bin_idx = np.clip(np.floor(phase * n_bins).astype(int), 0, n_bins - 1)
        np.add.at(bin_sums, bin_idx, flux_norm)
        np.add.at(bin_counts, bin_idx, 1)
        
        # Avoid division by zero
        valid = bin_counts > 0
        bin_flux = np.zeros(n_bins)
        bin_flux[valid] = bin_sums[valid] / bin_counts[valid]
        
        # Test different transit durations
        for dur_frac in duration_fracs:
            n_transit_bins = max(1, int(dur_frac * n_bins))
            
            # Slide transit window using convolution (fast)
            kernel = np.ones(n_transit_bins) / n_transit_bins
            transit_mean = np.convolve(bin_flux, kernel, mode='same')
            
            # Out-of-transit mean
            total_mean = np.mean(bin_flux)
            
            # Depth estimate at each phase
            depth_est = total_mean - transit_mean
            
            # Best depth for this period/duration
            best_idx = np.argmax(depth_est)
            if depth_est[best_idx] > 0:
                sr = depth_est[best_idx]**2 * n_transit_bins
                if sr > power[i]:
                    power[i] = sr
                
                if sr > max_power:
                    max_power = sr
                    best_params['period'] = period
                    best_params['depth'] = depth_est[best_idx]
                    best_params['t0'] = bin_edges[best_idx] * period
                    best_params['duration'] = dur_frac * period
    
    # Calculate SNR
    noise = median_abs_deviation(flux_norm) * 1.4826
    if noise > 0 and best_params['period'] > 0:
        best_params['snr'] = best_params['depth'] / noise * np.sqrt(
            len(time) * best_params['duration'] / best_params['period']
        )
    
    return {
        'periods': periods,
        'power': power / np.max(power) if np.max(power) > 0 else power,
        **best_params
    }


def generalized_lomb_scargle(time, flux, min_period=1.0, max_period=20.0, n_periods=500):
    """
    Optimized Generalized Lomb-Scargle periodogram.
    """
    periods = np.linspace(min_period, max_period, n_periods)
    frequencies = 1.0 / periods
    
    # Normalize flux
    flux_norm = flux - np.mean(flux)
    var = np.var(flux_norm)
    
    power = np.zeros(len(frequencies))
    
    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq
        wt = omega * time
        
        # Vectorized calculations
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)
        
        # Calculate tau
        tau = np.arctan2(np.sum(np.sin(2 * wt)), np.sum(np.cos(2 * wt))) / (2 * omega)
        
        wt_tau = omega * (time - tau)
        cos_term = np.cos(wt_tau)
        sin_term = np.sin(wt_tau)
        
        cc = np.sum(cos_term**2)
        ss = np.sum(sin_term**2)
        
        if cc > 0 and ss > 0:
            yc = np.sum(flux_norm * cos_term)
            ys = np.sum(flux_norm * sin_term)
            power[i] = 0.5 * (yc**2 / cc + ys**2 / ss)
    
    # Normalize
    if var > 0:
        power = power / var
    
    # Find best period
    best_idx = np.argmax(power)
    
    return {
        'periods': periods,
        'power': power / np.max(power) if np.max(power) > 0 else power,
        'period': periods[best_idx],
        'power_max': power[best_idx] / np.max(power) if np.max(power) > 0 else 0
    }


def phase_dispersion_minimization(time, flux, min_period=1.0, max_period=20.0, 
                                   n_periods=500, n_bins=10):
    """
    Optimized Phase Dispersion Minimization (PDM) algorithm.
    """
    periods = np.linspace(min_period, max_period, n_periods)
    theta = np.ones(len(periods))
    
    total_var = np.var(flux)
    if total_var == 0:
        return {'periods': periods, 'theta': theta, 'power': np.zeros(len(periods)), 
                'period': periods[0], 'theta_min': 1.0}
    
    for i, period in enumerate(periods):
        phase = (time % period) / period
        
        # Use digitize for fast binning
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.digitize(phase, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Calculate variance in each bin
        weighted_var = 0
        total_weight = 0
        
        for b in range(n_bins):
            mask = bin_idx == b
            n_in_bin = np.sum(mask)
            if n_in_bin > 1:
                bin_var = np.var(flux[mask])
                weighted_var += bin_var * (n_in_bin - 1)
                total_weight += n_in_bin - 1
        
        if total_weight > 0:
            theta[i] = weighted_var / total_weight / total_var
    
    # Best period has minimum theta
    best_idx = np.argmin(theta)
    
    return {
        'periods': periods,
        'theta': theta,
        'power': 1 - theta,
        'period': periods[best_idx],
        'theta_min': theta[best_idx]
    }


def find_transits_multiple_methods(time, flux, min_period=1.0, max_period=20.0,
                                   methods=['bls', 'gls', 'pdm']):
    """
    Run multiple period-finding methods and combine results.
    """
    results = {}
    
    if 'bls' in methods:
        results['bls'] = box_least_squares(time, flux, min_period, max_period)
    
    if 'gls' in methods:
        results['gls'] = generalized_lomb_scargle(time, flux, min_period, max_period)
    
    if 'pdm' in methods:
        results['pdm'] = phase_dispersion_minimization(time, flux, min_period, max_period)
    
    # Calculate consensus period (weighted average)
    periods = []
    weights = []
    
    if 'bls' in results:
        periods.append(results['bls']['period'])
        weights.append(results['bls'].get('snr', 1) if results['bls'].get('snr', 0) > 0 else 1)
    
    if 'gls' in results:
        periods.append(results['gls']['period'])
        weights.append(results['gls']['power_max'] * 10)
    
    if 'pdm' in results:
        periods.append(results['pdm']['period'])
        weights.append((1 - results['pdm']['theta_min']) * 10)
    
    if len(periods) > 0:
        consensus_period = np.average(periods, weights=weights)
        consensus_std = np.std(periods)
    else:
        consensus_period = 0
        consensus_std = 0
    
    results['consensus'] = {
        'period': consensus_period,
        'period_std': consensus_std,
        'periods': periods,
        'weights': weights
    }
    
    return results


def measure_transit_times(time, flux, period, t0, duration):
    """
    Measure individual transit times for TTV analysis.
    """
    n_transits = int((time[-1] - time[0]) / period) + 1
    transit_times = []
    transit_times_err = []
    observed_times = []
    epoch = []
    
    for i in range(n_transits):
        expected_tc = t0 + i * period
        
        # Window around expected transit
        window = 2 * duration
        mask = np.abs(time - expected_tc) < window
        
        if np.sum(mask) < 5:
            continue
        
        t_window = time[mask]
        f_window = flux[mask]
        
        # Find minimum (transit center)
        min_idx = np.argmin(f_window)
        observed_tc = t_window[min_idx]
        
        # Simple error estimate
        tc_err = duration / 10  # Simplified
        
        transit_times.append(expected_tc)
        observed_times.append(observed_tc)
        transit_times_err.append(tc_err)
        epoch.append(i)
    
    # Calculate O-C (observed minus calculated)
    oc = np.array(observed_times) - np.array(transit_times)
    
    # Detect significant TTVs
    oc_rms = np.std(oc) if len(oc) > 1 else 0
    ttvs_detected = oc_rms > 0.001  # More than ~1.5 minutes
    
    return {
        'epoch': np.array(epoch),
        'expected_times': np.array(transit_times),
        'observed_times': np.array(observed_times),
        'timing_errors': np.array(transit_times_err),
        'oc': oc,
        'oc_minutes': oc * 24 * 60,
        'rms_ttv': oc_rms,
        'rms_ttv_minutes': oc_rms * 24 * 60,
        'ttvs_detected': ttvs_detected
    }


def injection_recovery_test(time, n_injections=50, period_range=(1, 15), 
                           depth_range=(0.001, 0.02), noise_level=0.001):
    """
    Perform injection-recovery test to assess detection efficiency.
    """
    results = []
    
    for i in range(n_injections):
        # Random transit parameters
        true_period = np.random.uniform(*period_range)
        true_depth = np.random.uniform(*depth_range)
        true_t0 = np.random.uniform(0, true_period)
        
        # Generate signal
        flux = generate_transit_signal(time, period=true_period, t0=true_t0, 
                                       depth=true_depth)
        flux_noisy = add_noise(flux, noise_level=noise_level)
        
        # Try to detect (use fewer periods for speed)
        bls_result = box_least_squares(time, flux_noisy, 
                                       min_period=period_range[0], 
                                       max_period=period_range[1],
                                       n_periods=300)
        
        # Check if recovered
        period_match = np.abs(bls_result['period'] - true_period) / true_period < 0.02
        
        results.append({
            'true_period': true_period,
            'true_depth': true_depth,
            'true_t0': true_t0,
            'detected_period': bls_result['period'],
            'detected_depth': bls_result['depth'],
            'snr': bls_result['snr'],
            'recovered': period_match
        })
    
    df = pd.DataFrame(results)
    
    # Calculate recovery rate by depth
    depth_bins = np.linspace(depth_range[0], depth_range[1], 6)
    recovery_by_depth = []
    
    for j in range(len(depth_bins) - 1):
        mask = (df['true_depth'] >= depth_bins[j]) & (df['true_depth'] < depth_bins[j+1])
        if mask.sum() > 0:
            rate = df[mask]['recovered'].mean()
        else:
            rate = 0
        recovery_by_depth.append({
            'depth_min': depth_bins[j],
            'depth_max': depth_bins[j+1],
            'depth_mid': (depth_bins[j] + depth_bins[j+1]) / 2,
            'recovery_rate': rate,
            'n_samples': mask.sum()
        })
    
    return {
        'individual_results': df,
        'recovery_by_depth': pd.DataFrame(recovery_by_depth),
        'overall_recovery_rate': df['recovered'].mean(),
        'n_injections': n_injections
    }


def validate_transit_parameters(period, depth, duration, snr):
    """
    Validate transit parameters against physical constraints.
    """
    warnings = []
    is_valid = True
    
    # Period checks
    if period < 0.1:
        warnings.append("⚠️ Period < 0.1 days is extremely short")
        is_valid = False
    elif period < 0.5:
        warnings.append("⚠️ Very short period - possible ultra-short period planet")
    
    if period > 1000:
        warnings.append("⚠️ Period > 1000 days may have insufficient phase coverage")
    
    # Depth checks
    if depth < 0.0001:
        warnings.append("⚠️ Depth < 0.01% may be noise artifact")
        is_valid = False
    elif depth < 0.001:
        warnings.append("ℹ️ Shallow transit - Earth-sized or smaller around Sun-like star")
    
    if depth > 0.1:
        warnings.append("⚠️ Depth > 10% is unusually deep - possible grazing EB")
    
    if depth > 0.25:
        warnings.append("⚠️ Depth > 25% indicates eclipsing binary, not planet")
        is_valid = False
    
    # Duration checks
    if duration > 0.3:
        warnings.append("⚠️ Duration > 0.3 days is very long")
    
    if duration / period > 0.2:
        warnings.append("⚠️ Duration/Period ratio > 20% is physically unlikely")
        is_valid = False
    
    # SNR checks
    if snr < 3:
        warnings.append("⚠️ SNR < 3 - detection not significant")
        is_valid = False
    elif snr < 7:
        warnings.append("⚠️ SNR < 7 - marginal detection")
    elif snr > 100:
        warnings.append("✓ High SNR detection")
    
    # Estimate planet parameters
    rp_rs = np.sqrt(depth)
    rp_rearth = rp_rs * 109  # Assuming Sun-like star
    
    planet_type = "Unknown"
    if rp_rearth < 1.25:
        planet_type = "Terrestrial"
    elif rp_rearth < 2.0:
        planet_type = "Super-Earth"
    elif rp_rearth < 4.0:
        planet_type = "Sub-Neptune"
    elif rp_rearth < 10:
        planet_type = "Neptune-like"
    else:
        planet_type = "Gas Giant"
    
    return {
        'is_valid': is_valid,
        'warnings': warnings,
        'rp_rs': rp_rs,
        'rp_rearth_approx': rp_rearth,
        'planet_type': planet_type
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_light_curve_plot(time, flux, flux_model=None, title="Light Curve"):
    """Create interactive light curve plot."""
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=time, y=flux,
        mode='markers',
        name='Data',
        marker=dict(size=3, color='#6366f1', opacity=0.6),
        hovertemplate="Time: %{x:.4f}<br>Flux: %{y:.6f}<extra></extra>"
    ))
    
    # Model if provided
    if flux_model is not None:
        fig.add_trace(go.Scatter(
            x=time, y=flux_model,
            mode='lines',
            name='Model',
            line=dict(color='#00d4aa', width=2),
            hovertemplate="Time: %{x:.4f}<br>Model: %{y:.6f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Time (days)',
        yaxis_title='Relative Flux',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(18,18,26,0.8)',
            bordercolor='rgba(0,212,170,0.3)',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_phase_folded_plot(time, flux, period, t0, title="Phase-Folded Light Curve"):
    """Create phase-folded light curve plot."""
    # Calculate phase
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    
    # Sort by phase for cleaner plotting
    sort_idx = np.argsort(phase)
    
    # Bin the data
    n_bins = 100
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_flux = np.zeros(n_bins)
    binned_err = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.median(flux[mask])
            binned_err[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
        else:
            binned_flux[i] = np.nan
    
    fig = go.Figure()
    
    # Individual points
    fig.add_trace(go.Scatter(
        x=phase, y=flux,
        mode='markers',
        name='Data',
        marker=dict(size=2, color='#6366f1', opacity=0.3),
        hoverinfo='skip'
    ))
    
    # Binned data
    fig.add_trace(go.Scatter(
        x=bin_centers, y=binned_flux,
        mode='markers',
        name='Binned',
        marker=dict(size=8, color='#00d4aa'),
        error_y=dict(array=binned_err, color='#00d4aa', thickness=1),
        hovertemplate="Phase: %{x:.3f}<br>Flux: %{y:.6f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Orbital Phase',
        yaxis_title='Relative Flux',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        legend=dict(
            bgcolor='rgba(18,18,26,0.8)',
            bordercolor='rgba(0,212,170,0.3)',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False, range=[-0.15, 0.15])
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_periodogram_plot(results, method_name="BLS"):
    """Create periodogram plot."""
    if 'periods' not in results:
        return go.Figure()
    
    power_key = 'power' if 'power' in results else 'theta'
    power = results[power_key] if 'power' in results else 1 - results['theta']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['periods'], y=power,
        mode='lines',
        name=method_name,
        line=dict(color='#00d4aa', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.1)'
    ))
    
    # Mark best period
    if 'period' in results:
        best_idx = np.argmin(np.abs(results['periods'] - results['period']))
        fig.add_trace(go.Scatter(
            x=[results['period']], y=[power[best_idx]],
            mode='markers',
            name=f'Best: {results["period"]:.4f} d',
            marker=dict(size=12, color='#f59e0b', symbol='star')
        ))
    
    fig.update_layout(
        title=dict(text=f'{method_name} Periodogram', font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Period (days)',
        yaxis_title='Power',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        legend=dict(
            bgcolor='rgba(18,18,26,0.8)',
            bordercolor='rgba(0,212,170,0.3)',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_ttv_plot(ttv_result):
    """Create TTV O-C diagram."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ttv_result['epoch'],
        y=ttv_result['oc_minutes'],
        mode='markers+lines',
        name='O-C',
        marker=dict(size=10, color='#00d4aa'),
        line=dict(color='#00d4aa', width=1, dash='dot'),
        error_y=dict(
            array=ttv_result['timing_errors'] * 24 * 60,
            color='#00d4aa',
            thickness=1
        )
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='#6366f1', opacity=0.5)
    
    # RMS bands
    rms = ttv_result['rms_ttv_minutes']
    fig.add_hrect(y0=-rms, y1=rms, fillcolor='rgba(99,102,241,0.1)', 
                  line_width=0, annotation_text=f'±{rms:.2f} min RMS')
    
    fig.update_layout(
        title=dict(text='Transit Timing Variations (O-C)', font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Epoch',
        yaxis_title='O-C (minutes)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk')
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_injection_recovery_plot(recovery_df):
    """Create injection-recovery efficiency plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=recovery_df['depth_mid'] * 100,
        y=recovery_df['recovery_rate'] * 100,
        marker=dict(
            color=recovery_df['recovery_rate'],
            colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#22c55e']],
            line=dict(width=1, color='rgba(255,255,255,0.3)')
        ),
        text=[f'{r*100:.0f}%' for r in recovery_df['recovery_rate']],
        textposition='outside',
        hovertemplate="Depth: %{x:.2f}%<br>Recovery: %{y:.1f}%<extra></extra>"
    ))
    
    # 50% threshold line
    fig.add_hline(y=50, line_dash='dash', line_color='#f59e0b', 
                  annotation_text='50% threshold')
    
    fig.update_layout(
        title=dict(text='Injection-Recovery Efficiency', font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Transit Depth (%)',
        yaxis_title='Recovery Rate (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        yaxis=dict(range=[0, 110])
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with navigation and settings."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin: 0;">🌟</h1>
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; 
                   color: #f1f5f9; margin: 0.5rem 0;">TransitKit</h2>
        <p style="color: #94a3b8; font-size: 0.8rem;">v2.0 • Transit Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Navigation
    mode = st.sidebar.radio(
        "📍 Analysis Mode",
        ["🌟 Synthetic Transit", "🔬 Multi-Method Detection", 
         "⏱️ TTV Analysis", "📊 Injection-Recovery", "📖 Documentation"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Quick Settings
    st.sidebar.markdown("### ⚙️ Quick Settings")
    
    noise_level = st.sidebar.slider(
        "Noise Level (ppm)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Photometric noise in parts per million"
    ) / 1e6
    
    observation_days = st.sidebar.slider(
        "Observation Duration (days)",
        min_value=10,
        max_value=100,
        value=30,
        help="Total observing baseline"
    )
    
    cadence = st.sidebar.slider(
        "Cadence (minutes)",
        min_value=1,
        max_value=30,
        value=2,
        help="Time between observations"
    )
    
    st.sidebar.markdown("---")
    
    # Info
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; text-align: center;">
        <p><a href="https://github.com/arifsolmaz/transitkit" target="_blank" 
              style="color: #00d4aa; text-decoration: none;">
           📦 GitHub Repository
        </a></p>
        <p>Based on Mandel & Agol (2002)</p>
        <p>© 2025 TransitKit</p>
    </div>
    """, unsafe_allow_html=True)
    
    return mode, noise_level, observation_days, cadence


# ============================================================================
# MAIN APPLICATION PAGES
# ============================================================================

def page_synthetic_transit(noise_level, observation_days, cadence):
    """Synthetic transit generation page."""
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">🌟</span>
        <h2>Synthetic Transit Generator</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Generate synthetic transit light curves using the Mandel & Agol (2002) limb-darkened model.
        Adjust parameters to explore different planetary scenarios.
    </div>
    """, unsafe_allow_html=True)
    
    # Parameter inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🪐 Planet Parameters")
        period = st.slider("Orbital Period (days)", 1.0, 30.0, 5.0, 0.1)
        depth = st.slider("Transit Depth (%)", 0.1, 5.0, 1.0, 0.1) / 100
        t0 = st.slider("Transit Epoch (days)", 0.0, float(period), period/2, 0.1)
    
    with col2:
        st.markdown("#### 🔄 Orbital Parameters")
        duration = st.slider("Transit Duration (days)", 0.05, 0.5, 0.15, 0.01)
        inc = st.slider("Inclination (°)", 80.0, 90.0, 89.0, 0.1)
        a_rs = st.slider("a/R★", 5.0, 50.0, 15.0, 1.0)
    
    with col3:
        st.markdown("#### ⭐ Limb Darkening")
        u1 = st.slider("u₁ (linear)", 0.0, 1.0, 0.4, 0.05)
        u2 = st.slider("u₂ (quadratic)", -0.5, 0.5, 0.2, 0.05)
        stellar_var = st.slider("Stellar Variability", 0.0, 0.005, 0.0, 0.0005)
    
    # Generate data
    n_points = int(observation_days * 24 * 60 / cadence)
    time = np.linspace(0, observation_days, n_points)
    
    flux_clean = generate_transit_signal(time, period=period, t0=t0, depth=depth,
                                         duration=duration, u1=u1, u2=u2,
                                         inc=inc, a_rs=a_rs)
    flux_noisy = add_noise(flux_clean, noise_level=noise_level, stellar_var=stellar_var)
    
    # Metrics
    rp_rs = np.sqrt(depth)
    rp_rearth = rp_rs * 109  # Sun-like star
    n_transits = int(observation_days / period)
    expected_snr = depth / noise_level * np.sqrt(n_points * duration / period)
    
    st.markdown("---")
    
    # Display metrics
    cols = st.columns(5)
    metrics = [
        ("Period", f"{period:.2f}", "days"),
        ("Depth", f"{depth*100:.3f}", "%"),
        ("Rp/R★", f"{rp_rs:.4f}", ""),
        ("~Rp", f"{rp_rearth:.1f}", "R⊕"),
        ("Expected SNR", f"{expected_snr:.1f}", "")
    ]
    
    for col, (label, value, unit) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{value}<span class="unit">{unit}</span></div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Plots
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_lc = create_light_curve_plot(time, flux_noisy, flux_clean, 
                                         title="Full Light Curve")
        st.plotly_chart(fig_lc, use_container_width=True)
    
    with col2:
        fig_phase = create_phase_folded_plot(time, flux_noisy, period, t0,
                                             title=f"Phase-Folded (P={period:.2f}d)")
        st.plotly_chart(fig_phase, use_container_width=True)
    
    # Store in session state
    st.session_state['time'] = time
    st.session_state['flux'] = flux_noisy
    st.session_state['flux_clean'] = flux_clean
    st.session_state['true_period'] = period
    st.session_state['true_depth'] = depth
    st.session_state['true_t0'] = t0
    st.session_state['true_duration'] = duration


def page_multi_method(noise_level, observation_days, cadence):
    """Multi-method detection comparison page."""
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">🔬</span>
        <h2>Multi-Method Transit Detection</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Compare BLS (Box Least Squares), GLS (Generalized Lomb-Scargle), and PDM (Phase Dispersion Minimization)
        algorithms for transit period detection. Consensus weighting combines results.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data exists
    if 'time' not in st.session_state:
        st.warning("⚠️ Please generate synthetic data first in the 'Synthetic Transit' tab.")
        return
    
    time = st.session_state['time']
    flux = st.session_state['flux']
    true_period = st.session_state.get('true_period', 5.0)
    
    # Period search range
    col1, col2, col3 = st.columns(3)
    with col1:
        min_period = st.slider("Min Period (days)", 0.5, 5.0, 1.0, 0.1)
    with col2:
        max_period = st.slider("Max Period (days)", 10.0, 50.0, 20.0, 1.0)
    with col3:
        methods = st.multiselect("Methods", ['bls', 'gls', 'pdm'], 
                                 default=['bls', 'gls', 'pdm'])
    
    if st.button("🚀 Run Detection", type="primary"):
        progress = st.progress(0, text="Starting detection...")
        
        results = {}
        n_methods = len(methods)
        
        for idx, method in enumerate(methods):
            progress.progress((idx) / n_methods, text=f"Running {method.upper()}...")
            
            if method == 'bls':
                results['bls'] = box_least_squares(time, flux, min_period, max_period)
            elif method == 'gls':
                results['gls'] = generalized_lomb_scargle(time, flux, min_period, max_period)
            elif method == 'pdm':
                results['pdm'] = phase_dispersion_minimization(time, flux, min_period, max_period)
        
        progress.progress(0.9, text="Calculating consensus...")
        
        # Calculate consensus
        periods = []
        weights = []
        
        if 'bls' in results:
            periods.append(results['bls']['period'])
            weights.append(max(results['bls'].get('snr', 1), 1))
        if 'gls' in results:
            periods.append(results['gls']['period'])
            weights.append(results['gls']['power_max'] * 10 + 1)
        if 'pdm' in results:
            periods.append(results['pdm']['period'])
            weights.append((1 - results['pdm']['theta_min']) * 10 + 1)
        
        results['consensus'] = {
            'period': np.average(periods, weights=weights) if periods else 0,
            'period_std': np.std(periods) if len(periods) > 1 else 0,
            'periods': periods,
            'weights': weights
        }
        
        progress.progress(1.0, text="Done!")
        progress.empty()
        
        st.session_state['detection_results'] = results
        
        st.markdown("---")
        st.markdown("### 📊 Results Comparison")
        
        # Results cards
        cols = st.columns(len(methods) + 1)
        
        method_colors = {'bls': '#00d4aa', 'gls': '#7c3aed', 'pdm': '#f59e0b'}
        
        for col, method in zip(cols[:-1], methods):
            r = results[method]
            period_err = abs(r['period'] - true_period) / true_period * 100
            is_correct = period_err < 2
            
            col.markdown(f"""
            <div class="method-card {'best' if is_correct else ''}">
                <div class="method-name" style="color: {method_colors[method]};">
                    {method.upper()}
                </div>
                <div class="param-row">
                    <span class="param-name">Period</span>
                    <span class="param-value">{r['period']:.4f} d</span>
                </div>
                <div class="param-row">
                    <span class="param-name">Error</span>
                    <span class="param-value" style="color: {'#22c55e' if is_correct else '#ef4444'}">
                        {period_err:.2f}%
                    </span>
                </div>
                {'<div style="color: #22c55e; font-size: 0.8rem; margin-top: 0.5rem;">✓ Match</div>' if is_correct else ''}
            </div>
            """, unsafe_allow_html=True)
        
        # Consensus
        c = results['consensus']
        cols[-1].markdown(f"""
        <div class="method-card best">
            <div class="method-name" style="color: #f1f5f9;">CONSENSUS</div>
            <div class="param-row">
                <span class="param-name">Period</span>
                <span class="param-value">{c['period']:.4f} d</span>
            </div>
            <div class="param-row">
                <span class="param-name">Scatter</span>
                <span class="param-value">{c['period_std']:.4f} d</span>
            </div>
            <div style="color: #00d4aa; font-size: 0.8rem; margin-top: 0.5rem;">★ Weighted</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Periodograms
        st.markdown("### 📈 Periodograms")
        
        cols = st.columns(len(methods))
        for col, method in zip(cols, methods):
            with col:
                fig = create_periodogram_plot(results[method], method.upper())
                
                # Add true period marker
                fig.add_vline(x=true_period, line_dash='dash', line_color='#ef4444',
                             annotation_text=f'True: {true_period:.2f}d')
                
                st.plotly_chart(fig, use_container_width=True)


def page_ttv_analysis(noise_level, observation_days, cadence):
    """TTV analysis page."""
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">⏱️</span>
        <h2>Transit Timing Variations (TTV)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Measure individual transit times and detect Transit Timing Variations. TTVs can indicate
        gravitational perturbations from additional planets in the system.
    </div>
    """, unsafe_allow_html=True)
    
    if 'time' not in st.session_state:
        st.warning("⚠️ Please generate synthetic data first.")
        return
    
    time = st.session_state['time']
    flux = st.session_state['flux']
    true_period = st.session_state.get('true_period', 5.0)
    true_t0 = st.session_state.get('true_t0', 2.5)
    true_duration = st.session_state.get('true_duration', 0.15)
    
    col1, col2 = st.columns(2)
    with col1:
        ttv_amplitude = st.slider("Inject TTV Amplitude (minutes)", 0.0, 30.0, 0.0, 1.0)
    with col2:
        ttv_period_ratio = st.slider("TTV Super-period (×orbital)", 2.0, 10.0, 5.0, 0.5)
    
    # If TTV amplitude > 0, inject TTVs
    if ttv_amplitude > 0:
        # Re-generate with TTVs
        n_points = len(time)
        flux_ttv = np.ones(n_points)
        n_transits = int((time[-1] - time[0]) / true_period) + 1
        
        ttv_epochs = []
        ttv_shifts = []
        
        for i in range(n_transits):
            # TTV sinusoidal pattern
            ttv_shift = ttv_amplitude / (24 * 60) * np.sin(2 * np.pi * i / ttv_period_ratio)
            actual_tc = true_t0 + i * true_period + ttv_shift
            
            ttv_epochs.append(i)
            ttv_shifts.append(ttv_shift * 24 * 60)  # Convert to minutes
            
            # Add transit at shifted time
            dt = np.abs(time - actual_tc)
            in_transit = dt < true_duration / 2
            
            if np.any(in_transit):
                x = (time[in_transit] - actual_tc) / (true_duration / 2)
                mu = np.sqrt(1 - np.clip(x**2, 0, 1))
                ld_factor = 1 - 0.4 * (1 - mu) - 0.2 * (1 - mu)**2
                depth = st.session_state.get('true_depth', 0.01)
                flux_ttv[in_transit] = 1 - depth * ld_factor * (1 - x**2)
        
        flux_with_ttv = add_noise(flux_ttv, noise_level)
        st.session_state['flux'] = flux_with_ttv
        flux = flux_with_ttv
        
        st.success(f"✓ Injected TTVs with amplitude ±{ttv_amplitude:.1f} minutes")
    
    if st.button("📏 Measure Transit Times", type="primary"):
        with st.spinner("Measuring individual transit times..."):
            ttv_result = measure_transit_times(time, flux, true_period, true_t0, true_duration)
        
        st.session_state['ttv_result'] = ttv_result
        
        st.markdown("---")
        
        # TTV detection result
        if ttv_result['ttvs_detected']:
            st.markdown("""
            <div class="success-box">
                <strong>✓ TTVs Detected!</strong><br>
                Significant transit timing variations found. This could indicate additional planets.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>ℹ️ No significant TTVs</strong><br>
                Transit times are consistent with a constant period.
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        cols = st.columns(4)
        cols[0].metric("Transits Measured", len(ttv_result['epoch']))
        cols[1].metric("RMS TTV", f"{ttv_result['rms_ttv_minutes']:.2f} min")
        cols[2].metric("Max |O-C|", f"{np.max(np.abs(ttv_result['oc_minutes'])):.2f} min")
        cols[3].metric("TTVs Detected", "Yes ✓" if ttv_result['ttvs_detected'] else "No")
        
        # O-C plot
        fig = create_ttv_plot(ttv_result)
        
        # Add injected TTVs if applicable
        if ttv_amplitude > 0:
            fig.add_trace(go.Scatter(
                x=ttv_epochs[:len(ttv_result['epoch'])],
                y=ttv_shifts[:len(ttv_result['epoch'])],
                mode='lines',
                name='Injected TTV',
                line=dict(color='#ef4444', width=2, dash='dash')
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transit times table
        with st.expander("📋 Transit Times Table"):
            df = pd.DataFrame({
                'Epoch': ttv_result['epoch'],
                'Expected (d)': ttv_result['expected_times'],
                'Observed (d)': ttv_result['observed_times'],
                'O-C (min)': ttv_result['oc_minutes'],
                'Error (min)': ttv_result['timing_errors'] * 24 * 60
            })
            st.dataframe(df.round(4), use_container_width=True)


def page_injection_recovery(noise_level, observation_days, cadence):
    """Injection-recovery test page."""
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">📊</span>
        <h2>Injection-Recovery Testing</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Assess transit detection efficiency by injecting synthetic transits and attempting recovery.
        This determines the completeness of your search for different planet sizes.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_injections = st.slider("Number of Injections", 10, 100, 30, 5)
    with col2:
        period_range = st.slider("Period Range (days)", 1.0, 20.0, (1.0, 15.0))
    with col3:
        depth_range = st.slider("Depth Range (%)", 0.1, 3.0, (0.1, 2.0))
    
    depth_range = (depth_range[0]/100, depth_range[1]/100)
    
    if st.button("🔬 Run Injection-Recovery Test", type="primary"):
        # Generate time array
        n_points = int(observation_days * 24 * 60 / cadence)
        time = np.linspace(0, observation_days, n_points)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run test with progress updates
        results = []
        for i in range(n_injections):
            progress_bar.progress((i + 1) / n_injections)
            status_text.text(f"Testing injection {i+1}/{n_injections}...")
            
            # Random parameters
            true_period = np.random.uniform(*period_range)
            true_depth = np.random.uniform(*depth_range)
            true_t0 = np.random.uniform(0, true_period)
            
            # Generate and detect
            flux = generate_transit_signal(time, period=true_period, t0=true_t0, depth=true_depth)
            flux_noisy = add_noise(flux, noise_level=noise_level)
            
            bls_result = box_least_squares(time, flux_noisy, 
                                           min_period=period_range[0],
                                           max_period=period_range[1],
                                           n_periods=2000)
            
            period_match = np.abs(bls_result['period'] - true_period) / true_period < 0.02
            
            results.append({
                'true_period': true_period,
                'true_depth': true_depth,
                'detected_period': bls_result['period'],
                'detected_depth': bls_result['depth'],
                'snr': bls_result['snr'],
                'recovered': period_match
            })
        
        progress_bar.empty()
        status_text.empty()
        
        df = pd.DataFrame(results)
        
        # Calculate recovery by depth
        depth_bins = np.linspace(depth_range[0], depth_range[1], 6)
        recovery_data = []
        for j in range(len(depth_bins) - 1):
            mask = (df['true_depth'] >= depth_bins[j]) & (df['true_depth'] < depth_bins[j+1])
            rate = df[mask]['recovered'].mean() if mask.sum() > 0 else 0
            recovery_data.append({
                'depth_mid': (depth_bins[j] + depth_bins[j+1]) / 2,
                'recovery_rate': rate,
                'n_samples': mask.sum()
            })
        recovery_df = pd.DataFrame(recovery_data)
        
        st.session_state['injection_results'] = df
        st.session_state['recovery_df'] = recovery_df
        
        st.markdown("---")
        st.success(f"✓ Completed {n_injections} injection-recovery tests")
        
        # Summary metrics
        cols = st.columns(4)
        cols[0].metric("Total Injections", n_injections)
        cols[1].metric("Overall Recovery", f"{df['recovered'].mean()*100:.1f}%")
        cols[2].metric("Mean SNR", f"{df['snr'].mean():.1f}")
        cols[3].metric("Median SNR", f"{df['snr'].median():.1f}")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_injection_recovery_plot(recovery_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Depth vs SNR scatter
            fig = px.scatter(
                df, x='true_depth', y='snr',
                color='recovered',
                color_discrete_map={True: '#22c55e', False: '#ef4444'},
                title='Detection Results',
                labels={'true_depth': 'True Depth', 'snr': 'Detection SNR', 
                        'recovered': 'Recovered'}
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9')
            )
            fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)')
            fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            st.dataframe(df.round(4), use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="injection_recovery_results.csv",
                mime="text/csv"
            )


def page_documentation():
    """Documentation and help page."""
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">📖</span>
        <h2>TransitKit Documentation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Start", "🔧 API Reference", 
                                       "📚 Methods", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ### Installation
        
        ```bash
        pip install transitkit
        ```
        
        ### Basic Usage
        
        ```python
        import numpy as np
        from transitkit.core import (
            generate_transit_signal_mandel_agol,
            find_transits_bls_advanced,
            add_noise
        )

        # Generate synthetic data
        time = np.linspace(0, 30, 2000)
        flux = generate_transit_signal_mandel_agol(
            time, 
            period=5.0, 
            t0=2.5, 
            depth=0.01
        )
        flux_noisy = add_noise(flux, noise_level=0.001)

        # Detect transit
        result = find_transits_bls_advanced(time, flux_noisy)
        print(f"Detected period: {result['period']:.4f} days")
        ```
        
        ### Load TESS Data
        
        ```python
        from transitkit.io import load_tess_data_advanced

        lc_collection = load_tess_data_advanced("TIC 25155310", sectors=[1, 2])
        ```
        """)
    
    with tab2:
        st.markdown("""
        ### Core Functions
        
        | Function | Description |
        |----------|-------------|
        | `generate_transit_signal_mandel_agol()` | Generate limb-darkened transit model |
        | `find_transits_bls_advanced()` | Box Least Squares with SNR/FAP |
        | `find_transits_multiple_methods()` | Multi-method consensus detection |
        | `find_period_gls()` | Generalized Lomb-Scargle |
        | `find_period_pdm()` | Phase Dispersion Minimization |
        | `estimate_parameters_mcmc()` | MCMC parameter estimation |
        
        ### Analysis Functions
        
        | Function | Description |
        |----------|-------------|
        | `detrend_light_curve_gp()` | Gaussian Process detrending |
        | `remove_systematics_pca()` | PCA systematics removal |
        | `measure_transit_timing_variations()` | TTV measurement |
        | `fit_transit_time()` | Fit individual transit times |
        
        ### Validation Functions
        
        | Function | Description |
        |----------|-------------|
        | `validate_transit_parameters()` | Physical parameter validation |
        | `perform_injection_recovery_test()` | Detection efficiency test |
        | `calculate_detection_significance()` | Bootstrap significance |
        """)
    
    with tab3:
        st.markdown("""
        ### Box Least Squares (BLS)
        
        The BLS algorithm searches for periodic box-shaped dips in the light curve.
        It's optimized for detecting planetary transits with flat-bottomed profiles.
        
        **Key parameters:**
        - `min_period`, `max_period`: Search range
        - `n_periods`: Resolution of period grid
        - `duration_grid`: Trial transit durations
        
        ### Generalized Lomb-Scargle (GLS)
        
        An extension of the classical Lomb-Scargle periodogram that includes
        a floating mean. Best for sinusoidal signals but can detect transits.
        
        ### Phase Dispersion Minimization (PDM)
        
        A non-parametric method that minimizes the dispersion of phase-folded data.
        Works well for non-sinusoidal periodic signals including transits.
        
        ### Consensus Detection
        
        TransitKit combines results from multiple methods using weighted averaging.
        Weights are based on detection significance (SNR, power, etc.).
        """)
    
    with tab4:
        st.markdown("""
        ### Frequently Asked Questions
        
        **Q: What noise level should I use?**
        
        A: Typical values:
        - TESS: 200-500 ppm for bright stars
        - Kepler: 50-200 ppm
        - Ground-based: 1000-5000 ppm
        
        **Q: How do I interpret SNR?**
        
        A: General guidelines:
        - SNR < 3: Not significant
        - SNR 3-7: Marginal detection
        - SNR 7-10: Good detection
        - SNR > 10: Strong detection
        
        **Q: Why do methods give different periods?**
        
        A: Each method has different sensitivity to signal shapes.
        BLS is optimal for box-shaped transits, while GLS works better
        for sinusoidal variations. Use the consensus period.
        
        **Q: How do I cite TransitKit?**
        
        ```bibtex
        @software{transitkit,
          author = {Solmaz, Arif},
          title = {TransitKit: Professional Exoplanet Transit Analysis},
          year = {2025},
          url = {https://github.com/arifsolmaz/transitkit}
        }
        ```
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Render sidebar and get settings
    mode, noise_level, observation_days, cadence = render_sidebar()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 TransitKit</h1>
        <p class="subtitle">Professional Exoplanet Transit Light Curve Analysis</p>
        <span class="badge">v2.0 • Based on Mandel & Agol (2002)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to appropriate page
    if "Synthetic" in mode:
        page_synthetic_transit(noise_level, observation_days, cadence)
    elif "Multi-Method" in mode:
        page_multi_method(noise_level, observation_days, cadence)
    elif "TTV" in mode:
        page_ttv_analysis(noise_level, observation_days, cadence)
    elif "Injection" in mode:
        page_injection_recovery(noise_level, observation_days, cadence)
    elif "Documentation" in mode:
        page_documentation()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with <a href="https://github.com/arifsolmaz/transitkit">TransitKit</a> • 
           <a href="https://streamlit.io">Streamlit</a> • 
           <a href="https://plotly.com">Plotly</a></p>
        <p>© 2025 TransitKit | MIT License</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()