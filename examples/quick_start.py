#!/usr/bin/env python
"""
TransitKit v2.0 - Quick Start Example

This script demonstrates the core functionality of TransitKit:
1. Generate synthetic transit light curve
2. Detect transit using BLS
3. Multi-method period detection
4. TTV analysis
5. Parameter validation
6. Create publication-quality plot

Author: Arif Solmaz
"""

import numpy as np
import matplotlib.pyplot as plt

# Import TransitKit modules
import transitkit as tk
from transitkit.core import (
    TransitParameters,
    generate_transit_signal_mandel_agol,
    add_noise,
    find_transits_bls_advanced,
    find_transits_multiple_methods,
)
from transitkit.analysis import measure_transit_timing_variations
from transitkit.utils import calculate_snr, check_data_quality
from transitkit.validation import validate_transit_parameters
from transitkit.visualization import setup_publication_style

print(f"TransitKit v{tk.__version__}")
print("=" * 60)


# =============================================================================
# 1. Generate Synthetic Transit Light Curve
# =============================================================================
print("\n1. Generating synthetic transit light curve...")

np.random.seed(42)

# System parameters
true_period = 5.0  # days
true_t0 = 2.5  # mid-transit time
true_depth = 0.01  # 1% transit depth (Jupiter-like around Sun-like star)

# Generate time array (50 days of observations)
time = np.linspace(0, 50, 3000)

# Generate transit signal using Mandel & Agol model
flux = generate_transit_signal_mandel_agol(
    time,
    period=true_period,
    t0=true_t0,
    depth=true_depth,
)

# Add realistic noise (1 mmag = 0.001 in normalized flux)
flux_noisy = add_noise(flux, noise_level=0.001)

print(f"   Time span: {time[-1] - time[0]:.1f} days")
print(f"   Number of points: {len(time)}")
print(f"   True period: {true_period} days")
print(f"   True depth: {true_depth} ({true_depth*1e6:.0f} ppm)")


# =============================================================================
# 2. Detect Transit Using BLS
# =============================================================================
print("\n2. Detecting transit using Box Least Squares (BLS)...")

bls_result = find_transits_bls_advanced(
    time, flux_noisy,
    min_period=1.0,
    max_period=20.0,
    n_periods=10000
)

period_error = abs(bls_result["period"] - true_period) / true_period * 100
depth_error = abs(bls_result["depth"] - true_depth) / true_depth * 100

print(f"   Detected period: {bls_result['period']:.4f} days (error: {period_error:.2f}%)")
print(f"   Detected depth: {bls_result['depth']:.5f} (error: {depth_error:.1f}%)")
print(f"   Transit time (T0): {bls_result['t0']:.4f}")
print(f"   Duration: {bls_result['duration']*24:.2f} hours")
print(f"   SNR: {bls_result['snr']:.1f}")
print(f"   FAP: {bls_result['fap']:.2e}")


# =============================================================================
# 3. Multi-Method Period Detection
# =============================================================================
print("\n3. Running multi-method period detection...")

multi_result = find_transits_multiple_methods(
    time, flux_noisy,
    min_period=1.0,
    max_period=20.0,
    methods=["bls", "gls", "pdm"]
)

print(f"   BLS period: {multi_result['bls']['period']:.4f} days")
print(f"   GLS period: {multi_result['gls']['period']:.4f} days")
print(f"   PDM period: {multi_result['pdm']['period']:.4f} days")
print(f"   Consensus period: {multi_result['consensus']['period']:.4f} days")


# =============================================================================
# 4. Transit Timing Variations (TTV) Analysis
# =============================================================================
print("\n4. Measuring Transit Timing Variations...")

ttv_result = measure_transit_timing_variations(
    time, flux_noisy,
    period=bls_result["period"],
    t0=bls_result["t0"],
    duration=bls_result["duration"]
)

print(f"   TTVs detected: {ttv_result['ttvs_detected']}")
print(f"   Number of transits: {len(ttv_result['ttvs'])}")
print(f"   RMS TTV: {ttv_result['rms_ttv']*24*60:.2f} minutes")
print(f"   TTV p-value: {ttv_result['p_value']:.3f}")


# =============================================================================
# 5. Calculate SNR and Validate Parameters
# =============================================================================
print("\n5. Validating transit parameters...")

# Calculate SNR
snr = calculate_snr(
    time, flux_noisy,
    period=bls_result["period"],
    t0=bls_result["t0"],
    duration=bls_result["duration"]
)
print(f"   Transit SNR: {snr:.1f}")

# Check data quality
quality = check_data_quality(time, flux_noisy)
print(f"   Data quality:")
print(f"      Points: {quality['n_points']}")
print(f"      Has NaNs: {quality['has_nans']}")
print(f"      Sorted: {quality['is_sorted']}")

# Create parameters object
params = TransitParameters(
    period=bls_result["period"],
    period_err=bls_result["errors"]["period_err"],
    t0=bls_result["t0"],
    t0_err=bls_result["errors"]["t0_err"],
    duration=bls_result["duration"],
    duration_err=bls_result["errors"]["duration_err"],
    depth=bls_result["depth"],
    depth_err=bls_result["errors"]["depth_err"],
    snr=snr,
    fap=bls_result["fap"]
)

# Validate parameters
validation = validate_transit_parameters(params, time, flux_noisy)
passed = sum(1 for k, v in validation.items() if isinstance(v, bool) and v)
total = sum(1 for k, v in validation.items() if isinstance(v, bool))
print(f"   Validation: {passed}/{total} checks passed")


# =============================================================================
# 6. Create Publication-Quality Plot
# =============================================================================
print("\n6. Creating visualization...")

setup_publication_style(style="aas", dpi=150)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Full light curve
ax1 = axes[0, 0]
ax1.plot(time, flux_noisy, ".", color="gray", alpha=0.5, markersize=1, label="Data")
ax1.plot(time, flux, "-", color="red", linewidth=0.5, alpha=0.7, label="Model")

# Mark transit locations
n_transits = int((time.max() - bls_result["t0"]) / bls_result["period"]) + 2
for n in range(-1, n_transits):
    tc = bls_result["t0"] + n * bls_result["period"]
    if time.min() <= tc <= time.max():
        ax1.axvline(tc, color="blue", alpha=0.3, linestyle="--", linewidth=0.5)

ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Normalized Flux")
ax1.set_title("Full Light Curve")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: Phase-folded light curve
ax2 = axes[0, 1]
phase = ((time - bls_result["t0"]) / bls_result["period"]) % 1
phase = (phase + 0.5) % 1 - 0.5  # Center at 0

# Bin the phase-folded data
n_bins = 100
bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
binned_flux = []
binned_err = []
for i in range(n_bins):
    mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
    if np.sum(mask) > 0:
        binned_flux.append(np.median(flux_noisy[mask]))
        binned_err.append(np.std(flux_noisy[mask]) / np.sqrt(np.sum(mask)))
    else:
        binned_flux.append(np.nan)
        binned_err.append(np.nan)

ax2.plot(phase, flux_noisy, ".", color="gray", alpha=0.1, markersize=1)
ax2.errorbar(bin_centers, binned_flux, yerr=binned_err, fmt="o", 
             color="blue", markersize=3, capsize=2, label="Binned")

# Mark transit duration
half_dur = 0.5 * bls_result["duration"] / bls_result["period"]
ax2.axvline(-half_dur, color="red", linestyle="--", alpha=0.5)
ax2.axvline(half_dur, color="red", linestyle="--", alpha=0.5)

ax2.set_xlabel("Phase")
ax2.set_ylabel("Normalized Flux")
ax2.set_title("Phase-Folded Light Curve")
ax2.set_xlim(-0.15, 0.15)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: BLS Periodogram
ax3 = axes[1, 0]
periods = bls_result.get("all_periods", np.array([]))
powers = bls_result.get("all_powers", np.array([]))

if len(periods) > 0:
    ax3.plot(periods, powers, "-", color="navy", linewidth=0.5)
    ax3.axvline(bls_result["period"], color="red", linestyle="--", 
                label=f"Best: {bls_result['period']:.3f} d")
    ax3.axvline(true_period, color="green", linestyle=":", alpha=0.7,
                label=f"True: {true_period:.1f} d")
    ax3.set_xlabel("Period (days)")
    ax3.set_ylabel("BLS Power")
    ax3.set_title("BLS Periodogram")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    if periods.max() / periods.min() > 10:
        ax3.set_xscale("log")

# Panel 4: Results summary
ax4 = axes[1, 1]
ax4.axis("off")

summary_text = f"""
Transit Detection Results
{'='*40}

Orbital Parameters:
  Period: {bls_result['period']:.6f} ± {params.period_err:.6f} days
  T₀: {bls_result['t0']:.4f} ± {params.t0_err:.4f} days
  Duration: {bls_result['duration']*24:.2f} ± {params.duration_err*24:.2f} hours

Transit Parameters:
  Depth: {bls_result['depth']*1e6:.1f} ± {params.depth_err*1e6:.1f} ppm
  Rp/Rs: {np.sqrt(bls_result['depth']):.4f}

Detection Statistics:
  SNR: {snr:.1f}
  FAP: {bls_result['fap']:.2e}

TTV Analysis:
  RMS TTV: {ttv_result['rms_ttv']*24*60:.2f} minutes
  Detected: {ttv_result['ttvs_detected']}

Validation: {passed}/{total} checks passed
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
         fontsize=10, fontfamily="monospace",
         verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.suptitle(f"TransitKit v{tk.__version__} - Transit Analysis Example", 
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_file = "transit_analysis_example.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"   Figure saved to: {output_file}")

plt.show()

print("\n" + "=" * 60)
print("Quick start example completed successfully!")
print("=" * 60)
