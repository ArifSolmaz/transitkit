# TransitKit Documentation

**Professional Exoplanet Transit Light Curve Analysis Toolkit**

TransitKit is a comprehensive Python package for analyzing exoplanet transit light curves. It provides publication-quality tools for transit detection, parameter estimation, validation, and visualization.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

tutorials/basic_analysis
tutorials/tess_data
tutorials/ttv_analysis
tutorials/mcmc_fitting
tutorials/publication_plots
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/core
api/analysis
api/validation
api/io
api/utils
api/visualization
api/nea
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
changelog
```

## Features

### Core Analysis
- **Transit Signal Generation**: Mandel & Agol (2002) limb-darkened transit models
- **Period Detection**: Multiple methods (BLS, GLS, PDM) with consensus weighting
- **Parameter Estimation**: MCMC-based fitting with full uncertainty quantification
- **Transit Timing Variations**: Automatic TTV detection and analysis

### Data Handling
- **TESS/Kepler Support**: Native `lightkurve` integration
- **Ground-based Data**: Flexible I/O for various formats
- **NASA Exoplanet Archive**: Direct TAP queries

### Validation & Quality
- **Detection Significance**: Bootstrap FAP estimation
- **Odd-Even Tests**: Eclipse depth consistency checks
- **Injection-Recovery**: Detection efficiency assessment

## Quick Example

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
    time, period=5.0, depth=0.01
)
flux_noisy = add_noise(flux, noise_level=0.001)

# Detect transit
result = find_transits_bls_advanced(time, flux_noisy)
print(f"Detected period: {result['period']:.4f} days")
print(f"SNR: {result['snr']:.1f}")
```

## Installation

```bash
# Basic installation
pip install transitkit

# With all optional features
pip install transitkit[all]
```

## Citation

If you use TransitKit in your research, please cite:

```bibtex
@software{transitkit,
  author = {Solmaz, Arif},
  title = {TransitKit: Professional Exoplanet Transit Analysis Toolkit},
  year = {2024},
  url = {https://github.com/arifsolmaz/transitkit},
  version = {2.0.0}
}
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
