"""
TransitKit Example: Generate and analyze a transit signal
"""

import numpy as np
import matplotlib.pyplot as plt
import transitkit as tk

print("=" * 50)
print("TransitKit Example: Synthetic Transit Analysis")
print("=" * 50)

# 1. Generate time array
time = np.linspace(0, 30, 3000)  # 30 days, 3000 points
print(f"Generated time array: {len(time)} points")
print(f"Time range: {time[0]:.1f} to {time[-1]:.1f} days")

# 2. Generate transit signal
print("\nGenerating transit signal...")
transit_params = {
    'period': 5.0,      # 5-day orbit
    'depth': 0.02,      # 2% transit depth
    'duration': 0.15,   # 3.6-hour transit
}

clean_flux = tk.generate_transit_signal(time, **transit_params)

# 3. Add noise
print("Adding noise...")
noisy_flux = tk.add_noise(clean_flux, noise_level=0.001)

print(f"\nTransit Parameters:")
for key, value in transit_params.items():
    print(f"  {key}: {value}")

# 4. Plot
print("\nCreating plot...")
fig = tk.plot_light_curve(time, noisy_flux, "Synthetic Transit Light Curve")

# Add transit markers
period = transit_params['period']
t0 = period / 2  # First transit
n_transits = int(time[-1] / period)

for i in range(n_transits):
    transit_time = t0 + i * period
    plt.axvline(x=transit_time, color='r', alpha=0.3, linestyle='--', linewidth=1)

plt.savefig('example_transit.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'example_transit.png'")

# 5. Try to find transits
print("\nSearching for transits...")
results = tk.find_transits_box(time, noisy_flux, min_period=1.0, max_period=20.0)

print(f"\nSearch Results:")
print(f"  Best period: {results['period']:.3f} days")
print(f"  True period: {transit_params['period']:.3f} days")
print(f"  Difference: {abs(results['period'] - transit_params['period']):.3f} days")
print(f"  Detection score: {results['score']:.3f}")

# 6. Plot periodogram
plt.figure(figsize=(10, 4))
plt.plot(results['all_periods'], results['all_scores'], 'b-', linewidth=2)
plt.axvline(x=transit_params['period'], color='r', linestyle='--', label='True period')
plt.axvline(x=results['period'], color='g', linestyle='--', label='Detected period')
plt.xlabel('Period (days)')
plt.ylabel('Detection Score')
plt.title('Period Search Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('example_periodogram.png', dpi=150, bbox_inches='tight')

print("\nPeriodogram saved as 'example_periodogram.png'")
print("\n" + "=" * 50)
print("Example complete! Check the generated PNG files.")
print("=" * 50)

# Show plots
plt.show()