# Installation

## Basic Installation

```bash
pip install transitkit
```

## Full Installation (with all features)

```bash
pip install "transitkit[full]"
```

## Development Installation

```bash
git clone https://github.com/arifsolmaz/transitkit.git
cd transitkit
pip install -e ".[dev,full]"
```

## Dependencies

TransitKit requires:
- Python 3.9+
- NumPy, SciPy, Matplotlib
- Astropy, Lightkurve
- Pandas, Scikit-learn

## Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `mcmc` | emcee, corner | MCMC parameter estimation |
| `cli` | click, rich | Command-line interface |
| `full` | All above + batman | Full functionality |

## Verify Installation

```python
import transitkit
print(transitkit.__version__)
```
