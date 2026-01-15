# Transit Timing Variations (TTV)

Detect timing deviations that may indicate additional planets.

## What are TTVs?

Transit timing variations occur when gravitational perturbations from other planets cause the transit times to deviate from a linear ephemeris.

## Measuring TTVs

```python
from transitkit.analysis import measure_transit_timing_variations

result = measure_transit_timing_variations(
    time, flux,
    period=5.0,
    t0=2.5,
    duration=0.15
)

print(f"TTVs Detected: {result['ttvs_detected']}")
print(f"RMS TTV: {result['rms_ttv']*24*60:.2f} minutes")
```

## O-C Diagram

The O-C (Observed minus Calculated) diagram shows timing deviations:

```python
import matplotlib.pyplot as plt

epochs = result['epochs']
ttvs_minutes = np.array(result['ttvs']) * 24 * 60

plt.plot(epochs, ttvs_minutes, 'bo')
plt.axhline(0, color='gray', ls='--')
plt.xlabel('Epoch')
plt.ylabel('O-C (minutes)')
plt.title('Transit Timing Variations')
plt.show()
```
