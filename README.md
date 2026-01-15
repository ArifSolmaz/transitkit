# TransitKit v3.0 "Universal"

## 🎯 The Magic: ANY Planet, ONE Line

```python
from transitkit import UniversalTarget

# Literally ANY identifier works:
target = UniversalTarget("WASP-39 b")           # Planet name
target = UniversalTarget("TIC 374829238")       # TESS ID  
target = UniversalTarget("TOI-700 d")           # TOI
target = UniversalTarget("Kepler-442b")         # Kepler planet
target = UniversalTarget("KIC 8191672")         # Kepler star
target = UniversalTarget("KOI-7016.01")         # Kepler candidate
target = UniversalTarget("EPIC 201912552")      # K2 target
target = UniversalTarget("HD 209458 b")         # HD catalog
target = UniversalTarget("GJ 1214 b")           # Gliese catalog
target = UniversalTarget("TRAPPIST-1 e")        # Famous systems
target = UniversalTarget("Proxima Centauri b")  # Proper names
target = UniversalTarget("Gaia DR3 4050234234") # Gaia ID
target = UniversalTarget("2MASS J12345678")     # 2MASS
target = UniversalTarget("285.67 -32.54")       # RA/Dec coordinates

# Then just:
target.analyze()      # Full pipeline
target.export("paper/")  # Publication-ready
```

## 🌟 What It Does

**Input**: Any identifier you can think of  
**Output**: Complete cross-matched data + analysis + publication materials

```
🔍 Resolving: WASP-39 b

============================================================
✅ Resolved: WASP-39 b
============================================================

📋 Identifiers:
   TIC: 192971669
   Gaia DR3: 6718676574376532864
   2MASS: J14292360-0322375
   HD: --

⭐ Host Star:
   Teff: 5485 K
   Radius: 0.939 R☉
   Mass: 0.931 M☉
   Distance: 215.1 pc

🪐 Planets (1 found):
   WASP-39 b: P=4.0553d, Rp=12.63 R⊕

📡 Available Data:
   TESS: Sectors [14, 40, 41] (['2min'])
   JWST: Programs ['1366', '2512'] (['NIRSPEC', 'MIRI'])
   🌈 Transmission spectrum available!
============================================================
```

## 🚀 Features

### Universal Target Resolution
- Cross-matches across **15+ catalogs** automatically
- SIMBAD → NASA Exoplanet Archive → MAST → ExoFOP → TIC/KIC
- Returns ALL known identifiers for your target

### Multi-Mission Data Fusion
```python
# Get EVERYTHING available
data = target.get_lightcurves()

# Or be specific
tess_data = target.get_lightcurves(missions=['TESS'])
kepler_data = target.get_lightcurves(missions=['Kepler'])

# Stitch 10+ years of data together
combined = data.stitch()  # Kepler (2009) + K2 + TESS (2024)
print(f"Total baseline: {combined.total_timespan:.0f} days")
```

### JWST Spectroscopy
```python
# Automatic transmission spectrum retrieval
spectrum = target.get_transmission_spectrum()

# Molecule detection
from transitkit import JWSTSpectroscopy
jwst = JWSTSpectroscopy(target)
molecules = jwst.detect_molecules()

for m in molecules:
    if m.detected:
        print(f"  {m.molecule}: {m.significance:.1f}σ at {m.wavelength_range}")
```

### ML-Powered Detection
```python
from transitkit import MLTransitDetector

# Works on ANY light curve
detector = MLTransitDetector(target)
candidates = detector.detect(time, flux)

# Combines BLS + TLS + Neural Network
# Returns vetted candidates with FP probabilities
for c in candidates:
    print(f"P={c.period:.4f}d  SNR={c.snr:.1f}  ML={c.ml_score:.2f}  FP={c.fp_probability:.0%}")
```

### One-Click Publication
```python
# Generate everything for your paper
target.export("my_paper/")

# Creates:
# my_paper/
# ├── paper.tex           # Complete skeleton
# ├── tables/
# │   ├── stellar_params.tex
# │   └── planet_params.tex
# └── figures/
#     ├── lightcurve.pdf
#     ├── transit.pdf
#     └── transmission.pdf
```

## 📦 Installation

```bash
# Basic install
pip install transitkit

# Full install (ML + spectroscopy)
pip install transitkit[full]

# Development
pip install transitkit[dev]
```

### Dependencies

**Core** (auto-installed):
- numpy, astropy, astroquery, lightkurve, matplotlib

**Optional**:
- `transitleastsquares` - TLS detection
- `tensorflow` - ML classification  
- `petitRADTRANS` - Atmospheric retrieval
- `emcee` - MCMC fitting

## 🎓 Examples

### Example 1: Quick Look at Any Target
```python
from transitkit import quick_look

# Just see what's available
quick_look("TOI-700 d")
quick_look("TIC 259377017") 
quick_look("Kepler-62f")
```

### Example 2: Full Analysis Pipeline
```python
from transitkit import UniversalTarget

target = UniversalTarget("HAT-P-11 b")

# Run everything
results = target.analyze()

print(f"Light curves: {len(results['lightcurves'])}")
print(f"Candidates: {len(results['candidates'])}")
print(f"Best: P={results['candidates'][0].period:.4f}d")
```

### Example 3: Hunt for New Planets
```python
from transitkit import UniversalTarget, MLTransitDetector

# Pick any TIC ID
target = UniversalTarget("TIC 12345678")

# Download all data
lcs = target.get_lightcurves()

# Search for transits
detector = MLTransitDetector(target)
for lc in lcs:
    candidates = detector.detect(lc.time, lc.flux)
    for c in candidates:
        if c.ml_score > 0.8 and c.snr > 10:
            print(f"🎯 Candidate! P={c.period:.4f}d Rp={c.rp_earth:.1f} R⊕")
```

### Example 4: Multi-Mission Time Series
```python
from transitkit import UniversalTarget

# A target with Kepler + TESS data
target = UniversalTarget("Kepler-442b")

# Get all missions
data = target.get_lightcurves()

print(f"Kepler: {len([l for l in data if l.mission=='Kepler'])} quarters")
print(f"TESS: {len([l for l in data if l.mission=='TESS'])} sectors")

# Stitch together for TTV analysis
combined = data.stitch()
print(f"Baseline: {combined.total_timespan/365:.1f} years!")
```

### Example 5: JWST Atmospheric Analysis
```python
from transitkit import UniversalTarget, JWSTSpectroscopy

target = UniversalTarget("WASP-39 b")
jwst = JWSTSpectroscopy(target)

# Get transmission spectrum
spectrum = jwst.get_transmission_spectrum()

# Detect molecules
molecules = jwst.detect_molecules(spectrum)
print("Detected molecules:")
for m in molecules[:5]:
    if m.detected:
        print(f"  {m.molecule}: {m.significance:.1f}σ")

# Fit atmosphere (requires petitRADTRANS)
atm = jwst.fit_atmosphere(spectrum)
print(f"T_eq = {atm.temperature} K")
```

## 🔧 API Reference

### UniversalTarget

The main class - accepts ANY identifier.

```python
target = UniversalTarget(identifier, verbose=True)

# Properties
target.ids          # CrossMatchedIDs - all catalog IDs
target.stellar      # StellarParameters - host star
target.planets      # List[PlanetParameters] - known planets
target.available_data  # AvailableData - what missions have data
target.tic          # Quick access to TIC ID
target.kic          # Quick access to KIC ID
target.coords       # (RA, Dec) tuple

# Methods
target.analyze()              # Full pipeline
target.get_lightcurves()      # Download all LCs
target.get_transmission_spectrum()  # JWST spectrum
target.export(output_dir)     # Publication materials
target.to_dict()             # Export as dict
target.to_json(filepath)     # Export as JSON
```

### Supported Identifiers

| Format | Examples |
|--------|----------|
| Planet names | `WASP-39 b`, `HD 209458 b`, `GJ 1214 b` |
| TIC | `TIC 12345678`, `TIC12345678` |
| KIC | `KIC 8191672`, `Kepler-442` |
| TOI | `TOI-700`, `TOI-700.01`, `TOI-700 d` |
| KOI | `KOI-7016.01`, `K7016.01` |
| EPIC | `EPIC 201912552`, `K2-18 b` |
| HD | `HD 209458`, `HD 209458 b` |
| HIP | `HIP 12345` |
| Gaia | `Gaia DR3 12345678` |
| 2MASS | `2MASS J12345678+1234567` |
| Coordinates | `285.67 -32.54`, `19h12m34s -32d54m12s` |

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE).

## 📚 Citation

If you use TransitKit in your research:

```bibtex
@software{transitkit,
  author = {Solmaz, Arif},
  title = {TransitKit: Universal Exoplanet Transit Analysis},
  version = {3.0.0},
  url = {https://github.com/arifsolmaz/transitkit}
}
```

---

**TransitKit v3.0** - *Any planet. Any mission. One toolkit.*
