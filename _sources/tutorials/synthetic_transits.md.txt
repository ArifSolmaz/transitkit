# Synthetic Transit Generation

Learn how to create realistic synthetic transit signals.

## Basic Transit Model

```python
from transitkit.core import generate_transit_signal_mandel_agol

time = np.linspace(0, 10, 1000)
flux = generate_transit_signal_mandel_agol(
    time,
    period=3.0,
    t0=1.5,
    depth=0.015,
    duration=0.1
)
```

## Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `period` | Orbital period (days) | 0.5 - 100 |
| `t0` | Mid-transit time | Any |
| `depth` | Transit depth (Rp/Rs)² | 0.0001 - 0.03 |
| `duration` | Transit duration (days) | 0.02 - 0.5 |
| `u1`, `u2` | Limb darkening | 0.0 - 1.0 |

## Adding Noise

```python
from transitkit.core import add_noise

# Add Gaussian noise (1000 ppm)
flux_noisy = add_noise(flux, noise_level=0.001)

# With specific random seed for reproducibility
flux_noisy = add_noise(flux, noise_level=0.001, seed=42)
```
