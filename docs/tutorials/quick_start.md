# Quick Start Guide

This tutorial demonstrates the basic workflow for transit detection.

## Generate Synthetic Data

```python
import numpy as np
from transitkit.core import (
    generate_transit_signal_mandel_agol,
    find_transits_bls_advanced,
    add_noise
)

# Create a 30-day observation
time = np.linspace(0, 30, 2000)

# Generate transit signal
flux = generate_transit_signal_mandel_agol(
    time,
    period=5.0,      # 5-day orbital period
    t0=2.5,          # Mid-transit at day 2.5
    depth=0.01,      # 1% transit depth
    duration=0.15    # ~3.6 hour duration
)

# Add realistic noise
flux_noisy = add_noise(flux, noise_level=0.001)
```

## Detect the Transit

```python
result = find_transits_bls_advanced(time, flux_noisy)

print(f"Period:   {result['period']:.4f} days")
print(f"Depth:    {result['depth']*1e6:.0f} ppm")
print(f"Duration: {result['duration']*24:.1f} hours")
print(f"SNR:      {result['snr']:.1f}")
```

## Visualize Results

```python
import matplotlib.pyplot as plt

# Phase fold the data
phase = ((time - result['t0']) / result['period']) % 1
phase[phase > 0.5] -= 1

plt.figure(figsize=(10, 5))
plt.plot(phase, flux_noisy, 'k.', ms=2, alpha=0.3)
plt.xlabel('Orbital Phase')
plt.ylabel('Normalized Flux')
plt.xlim(-0.1, 0.1)
plt.title(f'Phase-Folded Transit (P={result["period"]:.4f}d)')
plt.show()
```

## Next Steps

- Learn about [Multi-Method Detection](multi_method.md)
- Explore [TTV Analysis](ttv_analysis.md)
- Create [Publication Plots](publication_plots.md)
