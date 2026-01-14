"""
TransitKit v2.0 Quick Start Example

This example demonstrates the new v2.0 API for exoplanet transit light curve analysis.
Features improved API, better error handling, and enhanced visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import transitkit as tk

print("=" * 60)
print("TransitKit v2.0 - Quick Start Example")
print("=" * 60)

# -------------------------
# 1) Generate synthetic data
# -------------------------
print("Generating synthetic transit data...")

# Create time array
time = np.linspace(0, 30, 3000)  # 30 days, 3000 points

# Define transit parameters
params = {
    'period': 5.0,          # orbital period in days
    't0': 2.5,              # transit epoch
    'rp': 0.1,              # planet-to-star radius ratio
    'a': 10.0,              # semi-major axis in stellar radii
    'inc': 89.0,            # inclination in degrees
    'ecc': 0.0,             # eccentricity
    'omega': 90.0,          # argument of periastron
    'u': [0.3, 0.2],        # limb-darkening coefficients
}

# Generate transit model using new v2.0 API
model = tk.TransitModel(**params)
clean_flux = model.evaluate(time)

# Add realistic noise
rng = np.random.default_rng(42)
noise_level = 0.001
flux = clean_flux + rng.normal(0, noise_level, len(time))

print(f"Generated {len(time)} data points")
print(f"Noise level: {noise_level * 100:.1f}%")

# -------------------------
# 2) Preprocess the data
# -------------------------
print("\nPreprocessing data...")

# Create LightCurve object (new in v2.0)
lc = tk.LightCurve(time=time, flux=flux, flux_err=np.full_like(flux, noise_level))

# Remove outliers
lc = lc.remove_outliers(sigma=5.0)

# Normalize flux
lc = lc.normalize()

print(f"Data points after preprocessing: {len(lc.time)}")

# -------------------------
# 3) Search for transits
# -------------------------
print("\nSearching for transits...")

# Use BLS periodogram (improved in v2.0)
bls = tk.BLS()
results = bls.search(
    time=lc.time,
    flux=lc.flux,
    flux_err=lc.flux_err,
    min_period=2.0,
    max_period=20.0,
    n_periods=5000
)

print("Top 3 transit candidates:")
for i in range(min(3, len(results['periods']))):
    print(f"  {i+1}. Period: {results['periods'][i]:.4f} days, "
          f"Depth: {results['depths'][i]*100:.3f}%, "
          f"SNR: {results['snrs'][i]:.2f}")

# -------------------------
# 4) Fit transit model
# -------------------------
print("\nFitting transit model...")

# Use best candidate
best_period = results['periods'][0]
best_t0 = results['t0s'][0]

# Initialize fitter with priors
fitter = tk.TransitFitter()

# Set priors
priors = {
    'period': tk.Prior('normal', mu=best_period, sigma=0.01),
    't0': tk.Prior('normal', mu=best_t0, sigma=0.01),
    'rp': tk.Prior('uniform', lower=0.08, upper=0.12),
    'a': tk.Prior('uniform', lower=8.0, upper=12.0),
    'inc': tk.Prior('uniform', lower=85.0, upper=90.0),
}

# Fit using MCMC (new in v2.0)
print("Running MCMC fit (this may take a moment)...")
fit_result = fitter.fit_mcmc(
    lightcurve=lc,
    priors=priors,
    n_walkers=32,
    n_steps=2000,
    n_burnin=500
)

print("\nFit results:")
for param, value in fit_result['parameters'].items():
    if param in fit_result['errors']:
        err = fit_result['errors'][param]
        print(f"  {param:6}: {value:.4f} ± {err:.4f}")

# -------------------------
# 5) Create visualizations
# -------------------------
print("\nCreating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Full light curve
ax = axes[0, 0]
ax.plot(lc.time, lc.flux, 'k.', alpha=0.5, ms=2)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Full Light Curve')
ax.grid(True, alpha=0.3)

# Mark detected transits
transit_times = best_t0 + best_period * np.arange(-3, 4)
for tt in transit_times:
    ax.axvline(tt, color='r', alpha=0.3, linestyle='--')

# Plot 2: Folded light curve
ax = axes[0, 1]
folded_time = tk.fold_time(lc.time, best_period, best_t0)
phase = np.linspace(-0.5, 0.5, 1000)
model_phase = tk.fold_time(phase, best_period, 0)  # Model phase

# Bin data for cleaner plot
binned = tk.bin_lightcurve(folded_time, lc.flux, bin_width=0.001)
ax.plot(binned['phase'], binned['flux'], 'k.', alpha=0.6, ms=4)

# Plot best fit model
folded_model = model.evaluate(phase + best_t0)
ax.plot(model_phase, folded_model, 'r-', linewidth=2, alpha=0.8)

ax.set_xlabel('Phase')
ax.set_ylabel('Normalized Flux')
ax.set_title(f'Folded Light Curve (P = {best_period:.4f} d)')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.15, 0.15)

# Plot 3: BLS periodogram
ax = axes[1, 0]
ax.plot(results['periods'], results['power'], 'b-', linewidth=1.5)
ax.axvline(best_period, color='r', linestyle='--', alpha=0.7, 
           label=f'Best: {best_period:.4f} d')
ax.axvline(params['period'], color='g', linestyle='--', alpha=0.7,
           label=f'True: {params["period"]:.1f} d')
ax.set_xlabel('Period (days)')
ax.set_ylabel('BLS Power')
ax.set_title('Periodogram')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Corner plot of MCMC samples
ax = axes[1, 1]
samples = fit_result['samples'][:, :3]  # First 3 parameters
labels = ['Period', 't0', 'rp']

# Create simple scatter correlation plot
for i in range(3):
    for j in range(i+1, 3):
        if i == 0 and j == 1:
            ax.scatter(samples[:, i], samples[:, j], alpha=0.1, s=1)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_title('Parameter Correlations')
            ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transitkit_v2_quickstart.png', dpi=150, bbox_inches='tight')
print("Saved: transitkit_v2_quickstart.png")

# -------------------------
# 6) Save results
# -------------------------
print("\nSaving results...")

# Save fit results to file (new in v2.0)
fit_result.save('transit_fit_results.json')
print("Saved: transit_fit_results.json")

# Generate summary report
report = f"""
Transit Analysis Report
=======================
Date: {np.datetime64('now')}
Data points: {len(lc.time)}
Transit detection SNR: {results['snrs'][0]:.2f}

Best-fit Parameters:
--------------------
Period: {fit_result['parameters']['period']:.6f} ± {fit_result['errors']['period']:.6f} days
T0: {fit_result['parameters']['t0']:.6f} ± {fit_result['errors']['t0']:.6f} BJD
Rp/Rs: {fit_result['parameters']['rp']:.5f} ± {fit_result['errors']['rp']:.5f}
Semi-major axis: {fit_result['parameters']['a']:.3f} ± {fit_result['errors']['a']:.3f} Rs
Inclination: {fit_result['parameters']['inc']:.2f} ± {fit_result['errors']['inc']:.2f} deg

Fit Quality:
------------
χ²: {fit_result['chisq']:.2f}
Reduced χ²: {fit_result['red_chisq']:.2f}
BIC: {fit_result['bic']:.2f}
MCMC acceptance fraction: {fit_result['acceptance_fraction']:.3f}
"""

with open('transit_analysis_report.txt', 'w') as f:
    f.write(report)

print("Saved: transit_analysis_report.txt")

print("\n" + "=" * 60)
print("Quick Start Example Complete!")
print("Generated files:")
print("  - transitkit_v2_quickstart.png")
print("  - transit_fit_results.json")
print("  - transit_analysis_report.txt")
print("=" * 60)

plt.show()