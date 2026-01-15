# visualization.py - Publication-quality plotting
"""
Publication-quality visualization tools for transit light curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import seaborn as sns

def setup_publication_style(fontsize=12, dpi=300, style='default'):
    """
    Set up matplotlib for publication-quality figures.
    """
    if style == 'nature':
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 10,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'patch.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
        })
    elif style == 'aas':
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.7,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'patch.linewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
        })
    else:
        plt.rcParams.update({
            'font.size': fontsize,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
        })


def plot_transit_summary(time, flux, params, residuals=None, figsize=(12, 10)):
    """
    Create comprehensive transit summary plot.
    
    Parameters
    ----------
    time : array
        Time array
    flux : array
        Flux array
    params : TransitParameters or dict
        Transit parameters
    residuals : array, optional
        Fit residuals
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    setup_publication_style(style='aas')
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])
    
    # Panel A: Full light curve with transit markers
    ax1 = plt.subplot(gs[0, :])
    plot_full_light_curve(ax1, time, flux, params)
    
    # Panel B: Phase-folded light curve
    ax2 = plt.subplot(gs[1, 0])
    plot_phase_folded(ax2, time, flux, params)
    
    # Panel C: Individual transit
    ax3 = plt.subplot(gs[1, 1])
    plot_individual_transit(ax3, time, flux, params)
    
    # Panel D: BLS periodogram
    ax4 = plt.subplot(gs[2, 0])
    # plot_periodogram(ax4, time, flux)  # You'd need periodogram data
    
    # Panel E: Residuals
    if residuals is not None:
        ax5 = plt.subplot(gs[2, 1])
        plot_residuals(ax5, time, residuals)
    
    plt.tight_layout()
    return fig


def plot_full_light_curve(ax, time, flux, params, 
                         color='k', alpha=0.6, ms=2):
    """
    Plot full light curve with transit markers.
    """
    # Plot data
    ax.plot(time, flux, '.', color=color, alpha=alpha, markersize=ms, 
            label='Data')
    
    # Add predicted transit markers
    if hasattr(params, 'period') and hasattr(params, 't0'):
        period = params.period
        t0 = params.t0
        
        # Calculate transit centers
        tmin, tmax = time.min(), time.max()
        n_min = int(np.floor((tmin - t0) / period)) - 1
        n_max = int(np.ceil((tmax - t0) / period)) + 1
        
        for n in range(n_min, n_max + 1):
            tc = t0 + n * period
            if tmin <= tc <= tmax:
                ax.axvline(tc, color='r', alpha=0.3, linestyle='--', 
                          linewidth=0.8)
    
    # Formatting
    ax.set_xlabel('Time (BJD - 2457000)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Full Light Curve')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add statistics
    stats_text = (f"P = {params.period:.6f} ± {params.period_err:.6f} d\n"
                  f"Depth = {params.depth*1e6:.1f} ± {params.depth_err*1e6:.1f} ppm\n"
                  f"SNR = {params.snr:.1f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return ax


def plot_phase_folded(ax, time, flux, params, bins=200, 
                     color='k', model_color='r'):
    """
    Plot phase-folded light curve with binned data and model.
    """
    # Phase fold
    period = params.period
    t0 = params.t0
    
    phase = ((time - t0) / period) % 1
    phase = (phase + 0.5) % 1 - 0.5  # Center at phase 0
    
    # Bin data
    phase_bins = np.linspace(-0.5, 0.5, bins + 1)
    phase_centers = 0.5 * (phase_bins[1:] + phase_bins[:-1])
    
    binned_flux = []
    binned_err = []
    
    for i in range(bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.sum(mask) > 0:
            binned_flux.append(np.median(flux[mask]))
            binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
        else:
            binned_flux.append(np.nan)
            binned_err.append(np.nan)
    
    binned_flux = np.array(binned_flux)
    binned_err = np.array(binned_err)
    
    # Plot
    ax.errorbar(phase_centers, binned_flux, yerr=binned_err, 
                fmt='.', color=color, alpha=0.8, markersize=6,
                capsize=2, label='Binned data')
    
    # Plot individual points (faded)
    ax.plot(phase, flux, '.', color=color, alpha=0.1, markersize=1,
            label='Individual points')
    
    # Add model if available
    if hasattr(params, 'duration'):
        # Simple box model
        half_width = 0.5 * params.duration / period
        x_model = np.linspace(-0.5, 0.5, 1000)
        y_model = np.ones_like(x_model)
        in_transit = (x_model > -half_width) & (x_model < half_width)
        y_model[in_transit] = 1 - params.depth
        
        ax.plot(x_model, y_model, '-', color=model_color, linewidth=2,
                label='Transit model')
    
    # Formatting
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Phase-folded Light Curve')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', fontsize=9)
    
    # Add transit duration markers
    if hasattr(params, 'duration'):
        ax.axvline(-half_width, color='r', alpha=0.5, linestyle='--', linewidth=1)
        ax.axvline(half_width, color='r', alpha=0.5, linestyle='--', linewidth=1)
    
    return ax


def plot_individual_transit(ax, time, flux, params, epoch=0, 
                           window_factor=3):
    """
    Plot individual transit with model fit.
    """
    period = params.period
    t0 = params.t0
    duration = params.duration
    
    # Calculate transit center for given epoch
    tc = t0 + epoch * period
    
    # Extract data around transit
    window = window_factor * duration
    mask = (time >= tc - window) & (time <= tc + window)
    
    if np.sum(mask) < 10:
        return ax
    
    t_transit = time[mask] - tc  # Center at 0
    f_transit = flux[mask]
    
    # Plot data
    ax.plot(t_transit, f_transit, '.', color='k', alpha=0.7, markersize=4)
    
    # Add model
    if hasattr(params, 'depth'):
        # Simple box model
        half_duration = duration / 2
        x_model = np.linspace(-window, window, 1000)
        y_model = np.ones_like(x_model)
        in_transit = (x_model > -half_duration) & (x_model < half_duration)
        y_model[in_transit] = 1 - params.depth
        
        ax.plot(x_model, y_model, '-', color='r', linewidth=2, 
                label='Model')
    
    # Formatting
    ax.set_xlabel('Time from Mid-transit (days)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Individual Transit (Epoch {epoch})')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', fontsize=9)
    
    # Add duration markers
    ax.axvline(-duration/2, color='r', alpha=0.5, linestyle='--', linewidth=1)
    ax.axvline(duration/2, color='r', alpha=0.5, linestyle='--', linewidth=1)
    
    return ax


def plot_transit_duration_scan(ax, time, flux, period, t0, 
                              durations, metrics, metric_name='SNR'):
    """
    Plot transit duration scan results.
    """
    ax.plot(durations * 24, metrics, '-', linewidth=2)
    ax.set_xlabel('Transit Duration (hours)')
    ax.set_ylabel(metric_name)
    ax.set_title('Transit Duration Scan')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Mark best duration
    best_idx = np.argmax(metrics)
    best_duration = durations[best_idx]
    ax.axvline(best_duration * 24, color='r', alpha=0.5, 
               linestyle='--', linewidth=1, 
               label=f'Best: {best_duration*24:.2f} h')
    
    ax.legend(loc='best')
    
    return ax


def plot_mcmc_corner(samples, param_names, truths=None, figsize=(10, 10)):
    """
    Plot MCMC corner plot for transit parameters.
    """
    import corner
    
    fig = corner.corner(samples, labels=param_names, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12},
                       truths=truths, truth_color='r',
                       fig=plt.figure(figsize=figsize))
    
    return fig


def plot_periodogram_comparison(ax, periods, powers_bls, powers_gls=None, 
                               powers_pdm=None, best_period=None):
    """
    Plot comparison of different periodogram methods.
    """
    ax.plot(periods, powers_bls, '-', label='BLS', linewidth=2)
    
    if powers_gls is not None:
        ax.plot(periods, powers_gls, '-', label='GLS', linewidth=1.5, alpha=0.7)
    
    if powers_pdm is not None:
        # PDM has different metric (lower is better)
        ax.plot(periods, powers_pdm, '-', label='PDM', linewidth=1.5, alpha=0.7)
    
    if best_period is not None:
        ax.axvline(best_period, color='r', linestyle='--', 
                  label=f'Best: {best_period:.4f} d', alpha=0.7)
    
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Power / Theta')
    ax.set_title('Periodogram Comparison')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best')
    
    # Log scale for x-axis if period range is large
    if periods.max() / periods.min() > 100:
        ax.set_xscale('log')
    
    return ax


def create_transit_report_figure(time, flux, params, filename=None, 
                               dpi=300, format='pdf'):
    """
    Create complete transit report figure for publication.
    """
    setup_publication_style(style='aas', dpi=dpi)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 3, height_ratios=[2, 1.5, 1, 1], 
                          width_ratios=[1, 1, 1])
    
    # 1. Full light curve (top row spanning 3 columns)
    ax1 = plt.subplot(gs[0, :])
    plot_full_light_curve(ax1, time, flux, params)
    
    # 2. Phase-folded light curve
    ax2 = plt.subplot(gs[1, 0])
    plot_phase_folded(ax2, time, flux, params)
    
    # 3. Individual transits (3 panels)
    ax3 = plt.subplot(gs[1, 1])
    plot_individual_transit(ax3, time, flux, params, epoch=0)
    
    ax4 = plt.subplot(gs[1, 2])
    plot_individual_transit(ax4, time, flux, params, epoch=1)
    
    # 4. Periodogram
    ax5 = plt.subplot(gs[2, 0])
    # plot_periodogram(ax5, ...)  # Would need periodogram data
    
    # 5. Duration scan
    ax6 = plt.subplot(gs[2, 1])
    # plot_transit_duration_scan(ax6, ...)  # Would need scan data
    
    # 6. TTVs if available
    ax7 = plt.subplot(gs[2, 2])
    # plot_ttvs(ax7, ...)  # Would need TTV data
    
    # 7. Parameter table
    ax8 = plt.subplot(gs[3, :])
    ax8.axis('off')
    create_parameter_table(ax8, params)
    
    plt.suptitle(f'Transit Analysis Report: {getattr(params, "name", "Unknown")}', 
                fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if filename:
        plt.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
    
    return fig


def create_parameter_table(ax, params):
    """
    Create parameter table for report.
    """
    if isinstance(params, dict):
        param_dict = params
    else:
        param_dict = params.to_dict()
    
    # Create table data
    table_data = []
    for key, value in param_dict.items():
        if key.startswith('_'):  # Skip private
            continue
        
        if isinstance(value, float):
            formatted = f"{value:.6g}"
        elif isinstance(value, bool):
            formatted = str(value)
        elif isinstance(value, dict):
            continue  # Skip nested dicts
        else:
            formatted = str(value)
        
        table_data.append([key.replace('_', ' ').title(), formatted])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', 
                     cellLoc='left', colWidths=[0.4, 0.6])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Add title
    ax.set_title('Fitted Parameters', fontsize=11, pad=20)
    
    return ax