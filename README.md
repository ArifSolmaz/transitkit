# TransitKit v2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/arifsolmaz/transitkit/actions/workflows/tests.yml/badge.svg)](https://github.com/arifsolmaz/transitkit/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Professional Exoplanet Transit Light Curve Analysis Toolkit**

TransitKit is a comprehensive Python package for analyzing exoplanet transit light curves. It provides publication-quality tools for transit detection, parameter estimation, validation, and visualization.

## Features

### Core Analysis
- **Transit Signal Generation**: Mandel & Agol (2002) limb-darkened transit models via `batman`
- **Period Detection**: Multiple methods (BLS, GLS, PDM) with consensus weighting
- **Parameter Estimation**: MCMC-based fitting with full uncertainty quantification
- **Transit Timing Variations**: Automatic TTV detection and analysis

### Data Handling
- **TESS/Kepler Support**: Native `lightkurve` integration for space-based data
- **Ground-based Data**: Flexible I/O for various formats (CSV, FITS, HDF5)
- **NASA Exoplanet Archive**: Direct TAP queries for known planet parameters

### Validation & Quality
- **Detection Significance**: Bootstrap FAP estimation
- **Odd-Even Tests**: Eclipse depth consistency checks
- **Injection-Recovery**: Detection efficiency assessment
- **Secondary Eclipse**: False positive screening

### Visualization
- **Publication-Quality Plots**: AAS/Nature journal styles
- **Interactive GUI**: Full-featured Tkinter application
- **MCMC Diagnostics**: Corner plots and chain visualization

## Installation

### Basic Installation
```bash
pip install transitkit
```

### With All Features
```bash
pip install "transitkit[full]"
```

### Development Installation
```bash
git clone https://github.com/arifsolmaz/transitkit.git
cd transitkit
pip install -e ".[dev,full]"
```

### Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `mcmc` | emcee, corner | MCMC parameter estimation |
| `cli` | click, rich | Command-line interface |
| `full` | All above + batman | Full functionality |
| `dev` | pytest, black, etc. | Development |
| `docs` | sphinx, etc. | Documentation building |

## Interactive Demo App

TransitKit includes a Streamlit web app for interactive exploration:

```bash
pip install streamlit
streamlit run app.py
```

**Features:**
- 🌟 **Synthetic Transit** - Generate custom transits with adjustable parameters
- 🔬 **Multi-Method** - Compare BLS, GLS, PDM detection algorithms  
- ⏱️ **TTV Analysis** - Explore transit timing variations
- 📊 **Batch Analysis** - Run injection-recovery tests

![Demo App](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)

## Quick Start

### Basic Transit Detection

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
    time, 
    period=5.0, 
    t0=2.5, 
    depth=0.01  # 1% transit depth
)
flux_noisy = add_noise(flux, noise_level=0.001)

# Detect transit
result = find_transits_bls_advanced(time, flux_noisy)
print(f"Detected period: {result['period']:.4f} days")
print(f"Transit depth: {result['depth']:.5f}")
print(f"SNR: {result['snr']:.1f}")
```

### Load TESS Data

```python
from transitkit.io import load_tess_data_advanced

# Load TESS data for a known planet
lc_collection = load_tess_data_advanced("TIC 25155310", sectors=[1, 2])

# Process each sector
for lc in lc_collection:
    time = lc.time.value
    flux = lc.flux.value
    # Analyze...
```

### Multi-Method Analysis

```python
from transitkit.core import find_transits_multiple_methods

# Use BLS, GLS, and PDM together
results = find_transits_multiple_methods(
    time, flux,
    min_period=1.0,
    max_period=20.0,
    methods=["bls", "gls", "pdm"]
)

print(f"BLS period: {results['bls']['period']:.4f} d")
print(f"GLS period: {results['gls']['period']:.4f} d")
print(f"PDM period: {results['pdm']['period']:.4f} d")
print(f"Consensus: {results['consensus']['period']:.4f} d")
```

### TTV Analysis

```python
from transitkit.analysis import measure_transit_timing_variations

ttv_result = measure_transit_timing_variations(
    time, flux,
    period=5.0,
    t0=2.5,
    duration=0.15
)

print(f"TTVs detected: {ttv_result['ttvs_detected']}")
print(f"RMS TTV: {ttv_result['rms_ttv']*24*60:.2f} minutes")
```

### Publication-Quality Plots

```python
from transitkit.visualization import create_transit_report_figure, setup_publication_style
from transitkit.core import TransitParameters

params = TransitParameters(
    period=5.0, period_err=0.001,
    t0=2.5, t0_err=0.01,
    duration=0.15, duration_err=0.01,
    depth=0.01, depth_err=0.001,
    snr=50.0
)

setup_publication_style(style='aas', dpi=300)
fig = create_transit_report_figure(time, flux, params)
fig.savefig('transit_report.pdf', bbox_inches='tight')
```

## Command-Line Interface

> **Note:** CLI requires the `cli` extra: `pip install "transitkit[cli]"`

```bash
# Get version info
transitkit version

# Run BLS detection
transitkit analyze detect lightcurve.csv --method bls --min-period 1 --max-period 20
```

## GUI Application

```bash
# Launch the GUI (requires: pip install "transitkit[full]")
python -m transitkit.gui_app
```

Or from Python:
```python
from transitkit.gui_app import main
main()
```

## API Reference

### Core Module (`transitkit.core`)

| Function | Description |
|----------|-------------|
| `generate_transit_signal_mandel_agol()` | Generate limb-darkened transit model |
| `find_transits_bls_advanced()` | Box Least Squares with SNR/FAP |
| `find_transits_multiple_methods()` | Multi-method consensus detection |
| `find_period_gls()` | Generalized Lomb-Scargle |
| `find_period_pdm()` | Phase Dispersion Minimization |
| `estimate_parameters_mcmc()` | MCMC parameter estimation |
| `add_noise()` | Add Gaussian noise to flux |

### Analysis Module (`transitkit.analysis`)

| Function | Description |
|----------|-------------|
| `detrend_light_curve_gp()` | Gaussian Process detrending |
| `remove_systematics_pca()` | PCA systematics removal |
| `measure_transit_timing_variations()` | TTV measurement |
| `fit_transit_time()` | Fit individual transit times |

### Validation Module (`transitkit.validation`)

| Function | Description |
|----------|-------------|
| `validate_transit_parameters()` | Physical parameter validation |
| `perform_injection_recovery_test()` | Detection efficiency test |
| `calculate_detection_significance()` | Bootstrap significance |
| `validate_against_secondary_eclipse()` | Secondary eclipse check |

### I/O Module (`transitkit.io`)

| Function | Description |
|----------|-------------|
| `load_tess_data_advanced()` | Load TESS light curves |
| `load_kepler_data()` | Load Kepler/K2 data |
| `load_ground_based_data()` | Load ground-based data |
| `export_transit_results()` | Export results (JSON/CSV/HDF5) |

## Citation

If you use TransitKit in your research, please cite:

```bibtex
@software{transitkit,
  author = {Solmaz, Arif},
  title = {TransitKit: Professional Exoplanet Transit Analysis Toolkit},
  year = {2025},
  url = {https://github.com/arifsolmaz/transitkit},
  version = {2.0.0}
}
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [batman](https://github.com/lkreidberg/batman) for transit models
- [lightkurve](https://github.com/lightkurve/lightkurve) for TESS/Kepler data access
- [astropy](https://www.astropy.org/) for astronomical utilities
- [emcee](https://github.com/dfm/emcee) for MCMC sampling

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
