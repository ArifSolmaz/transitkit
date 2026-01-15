# 🌟 TransitKit Streamlit App

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Interactive web application for exoplanet transit light curve analysis.**

This Streamlit app provides a user-friendly interface to the [TransitKit](https://github.com/arifsolmaz/transitkit) Python package, enabling interactive exploration of transit detection, parameter estimation, and validation.

## ✨ Features

### 🌟 Synthetic Transit Generator
- Generate realistic transit light curves using Mandel & Agol (2002) limb-darkened models
- Adjustable planet, orbital, and stellar parameters
- Configurable noise levels and stellar variability
- Real-time visualization of full and phase-folded light curves

### 🔬 Multi-Method Detection
- **BLS** (Box Least Squares) - Optimized for box-shaped transits
- **GLS** (Generalized Lomb-Scargle) - Classical periodogram analysis
- **PDM** (Phase Dispersion Minimization) - Non-parametric method
- Consensus period combining all methods with weighted averaging

### ⏱️ TTV Analysis
- Measure individual transit times
- Detect Transit Timing Variations (TTVs)
- O-C (Observed minus Calculated) diagrams
- Inject synthetic TTVs for testing

### 📊 Injection-Recovery Testing
- Assess detection completeness
- Recovery efficiency as function of transit depth
- Statistical analysis of detection limits
- Export results for further analysis

## 🚀 Quick Start

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the forked repository
5. Set main file path to `app.py`
6. Click **Deploy**

### Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/transitkit-streamlit.git
cd transitkit-streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
transitkit-streamlit/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .streamlit/
    └── config.toml           # Streamlit configuration & theme
```

## 🎨 Screenshots

### Synthetic Transit Generator
Generate and visualize limb-darkened transit models with customizable parameters.

### Multi-Method Detection
Compare BLS, GLS, and PDM algorithms side-by-side with interactive periodograms.

### TTV Analysis
Measure transit timing variations and detect gravitational perturbations.

## 📚 Scientific Background

### Transit Model
The app uses the Mandel & Agol (2002) quadratic limb-darkening model:

```
I(μ) = 1 - u₁(1-μ) - u₂(1-μ)²
```

where μ = cos(θ) is the cosine of the angle between the line of sight and the normal to the stellar surface.

### Detection Methods

| Method | Best For | Reference |
|--------|----------|-----------|
| BLS | Box-shaped transits | Kovács et al. (2002) |
| GLS | Sinusoidal signals | Zechmeister & Kürster (2009) |
| PDM | Non-sinusoidal periodic signals | Stellingwerf (1978) |

### Planet Classification

| Type | Radius (R⊕) |
|------|-------------|
| Terrestrial | < 1.25 |
| Super-Earth | 1.25 - 2.0 |
| Sub-Neptune | 2.0 - 4.0 |
| Neptune-like | 4.0 - 10 |
| Gas Giant | > 10 |

## 🔗 Related

- [TransitKit Python Package](https://github.com/arifsolmaz/transitkit)
- [batman](https://github.com/lkreidberg/batman) - Transit model library
- [lightkurve](https://github.com/lightkurve/lightkurve) - TESS/Kepler data access

## 📝 Citation

If you use this application in your research, please cite:

```bibtex
@software{transitkit,
  author = {Solmaz, Arif},
  title = {TransitKit: Professional Exoplanet Transit Analysis Toolkit},
  year = {2025},
  url = {https://github.com/arifsolmaz/transitkit},
  version = {2.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*"The nitrogen in our DNA, the calcium in our teeth, the iron in our blood, the carbon in our apple pies were made in the interiors of collapsing stars. We are made of starstuff."* - Carl Sagan

🌟 Happy Transit Hunting! 🚀
