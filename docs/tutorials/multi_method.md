# Multi-Method Detection

Compare BLS, GLS, and PDM detection algorithms.

## Why Multiple Methods?

Different algorithms have different strengths:
- **BLS**: Best for box-shaped transits
- **GLS**: Good for sinusoidal variations
- **PDM**: Robust for irregular signals

## Running Multi-Method Detection

```python
from transitkit.core import find_transits_multiple_methods

results = find_transits_multiple_methods(
    time, flux,
    min_period=1.0,
    max_period=20.0,
    methods=['bls', 'gls', 'pdm']
)

print(f"BLS:       {results['bls']['period']:.4f} d")
print(f"GLS:       {results['gls']['period']:.4f} d")
print(f"PDM:       {results['pdm']['period']:.4f} d")
print(f"Consensus: {results['consensus']['period']:.4f} d")
```

## Consensus Period

The consensus combines all methods with weighted averaging based on detection significance.
