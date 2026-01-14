"""
TransitKit Example: Generate and analyze a transit signal (improved)

- Reproducible noise (fixed RNG seed)
- Robust period recovery using Astropy BoxLeastSquares (BLS) via tk.find_transits_box()
  (Assumes you replaced tk.find_transits_box with the astropy-based version I gave earlier.)
- Cleaner plotting: zoomed transit view + BLS power plot
- Better annotations and saved outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import transitkit as tk

print("=" * 60)
print("TransitKit Example: Synthetic Transit Analysis (BLS)")
print("=" * 60)

# -------------------------
# 0) Reproducibility
# -------------------------
SEED = 42
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")

# -------------------------
# 1) Generate time array
# -------------------------
t_start, t_stop, n_points = 0.0, 30.0, 3000
time = np.linspace(t_start, t_stop, n_points)
print(f"Generated time array: {len(time)} points")
print(f"Time range: {time[0]:.1f} to {time[-1]:.1f} days")
print(f"Cadence: ~{(time[1] - time[0]) * 24 * 60:.2f} minutes")

# -------------------------
# 2) Generate transit signal
# -------------------------
transit_params = {
    "period": 5.0,      # days
    "depth": 0.02,      # fractional (2%)
    "duration": 0.15,   # days (3.6 hours)
}

print("\nTransit Parameters:")
for k, v in transit_params.items():
    print(f"  {k:>8}: {v}")

print("\nGenerating synthetic transit signal...")
clean_flux = tk.generate_transit_signal(time, **transit_params)

# -------------------------
# 3) Add noise (reproducible)
# -------------------------
noise_level = 0.001
print(f"Adding Gaussian noise: sigma={noise_level}")
noisy_flux = clean_flux + rng.normal(0.0, noise_level, size=clean_flux.size)

# -------------------------
# 4) Plot full light curve + transit markers
# -------------------------
print("\nCreating light-curve plot...")
fig1 = tk.plot_light_curve(time, noisy_flux, "Synthetic Transit Light Curve (Noisy)")

# Add vertical markers at expected transit centers (for reference)
period = transit_params["period"]
t0 = period / 2.0  # consistent with tk.generate_transit_signal()
n_transits = int(time[-1] / period) + 1

for i in range(n_transits):
    transit_time = t0 + i * period
    plt.axvline(x=transit_time, color="r", alpha=0.25, linestyle="--", linewidth=1)

plt.savefig("example_transit.png", dpi=150, bbox_inches="tight")
print("Saved: example_transit.png")

# Optional: zoom panel around one transit for visibility
print("Creating zoomed transit view...")
zoom_center = t0 + 2 * period
zoom_half_width = 0.6  # days
mask = (time > zoom_center - zoom_half_width) & (time < zoom_center + zoom_half_width)

plt.figure(figsize=(10, 4))
plt.plot(time[mask], noisy_flux[mask], "k.", alpha=0.6, markersize=3)
plt.axvline(zoom_center, color="r", alpha=0.35, linestyle="--", linewidth=1)
plt.xlabel("Time (days)")
plt.ylabel("Normalized Flux")
plt.title("Zoomed View Around One Transit")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("example_transit_zoom.png", dpi=150, bbox_inches="tight")
print("Saved: example_transit_zoom.png")

# -------------------------
# 5) Period search (BLS)
# -------------------------
print("\nSearching for transits (BLS)...")
minP, maxP = 1.0, 20.0

# If your tk.find_transits_box is the astropy-BLS version:
results = tk.find_transits_box(
    time,
    noisy_flux,
    min_period=minP,
    max_period=maxP,
    # Optional tuning knobs if you exposed them in your BLS wrapper:
    # durations=np.linspace(0.05, 0.25, 20),
    # n_periods=5000,
)

bestP = results["period"]
trueP = transit_params["period"]
diff = abs(bestP - trueP)

print("\nSearch Results:")
print(f"  Best period   : {bestP:.6f} days")
print(f"  True period   : {trueP:.6f} days")
print(f"  |Difference|  : {diff:.6f} days")

# BLS-style outputs (if present)
if "t0" in results:
    print(f"  Best t0       : {results['t0']:.6f} days")
if "duration" in results:
    print(f"  Best duration : {results['duration']:.6f} days")
if "depth" in results:
    print(f"  Best depth    : {results['depth']:.6f}")
if results.get("snr", None) is not None:
    print(f"  SNR           : {results['snr']:.3f}")
else:
    # fallback (older wrapper)
    if "score" in results:
        print(f"  Score         : {results['score']:.3f}")

# -------------------------
# 6) Plot “periodogram” (BLS power)
# -------------------------
print("\nCreating BLS power plot...")
plt.figure(figsize=(10, 4))

# Handle both naming conventions robustly
all_periods = results.get("all_periods", None)
all_power = results.get("all_power", None)
all_scores = results.get("all_scores", None)

if all_periods is None:
    raise KeyError("results must include 'all_periods' for plotting.")

y = all_power if all_power is not None else all_scores
ylabel = "BLS Power" if all_power is not None else "Detection Score"

plt.plot(all_periods, y, linewidth=2)

plt.axvline(x=trueP, color="r", linestyle="--", label=f"True period = {trueP:.3f} d")
plt.axvline(x=bestP, color="g", linestyle="--", label=f"Detected = {bestP:.3f} d")

plt.xlabel("Period (days)")
plt.ylabel(ylabel)
plt.title("Period Search Results")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("example_periodogram.png", dpi=150, bbox_inches="tight")
print("Saved: example_periodogram.png")

print("\n" + "=" * 60)
print("Example complete! Check the generated PNG files:")
print("  - example_transit.png")
print("  - example_transit_zoom.png")
print("  - example_periodogram.png")
print("=" * 60)

plt.show()
