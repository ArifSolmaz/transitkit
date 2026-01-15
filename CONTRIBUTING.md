# Contributing to TransitKit

Thank you for your interest in contributing to TransitKit! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Your environment (Python version, OS, TransitKit version)
5. Any relevant code snippets or error messages

### Suggesting Features

Feature requests are welcome! Please open an issue with:

1. A clear description of the feature
2. The scientific use case it addresses
3. Any relevant papers or references
4. Example code showing how it might be used

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,all]"
   pre-commit install
   ```
3. **Make your changes** following our coding standards
4. **Add tests** for any new functionality
5. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```
6. **Update documentation** if needed
7. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/transitkit.git
cd transitkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/transitkit --cov-report=html

# Run specific test file
pytest tests/test_basic.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting

Run formatters before committing:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

Or let pre-commit handle it:

```bash
pre-commit run --all-files
```

### Documentation

Documentation is built with Sphinx:

```bash
cd docs
make html
```

View at `docs/_build/html/index.html`

## Coding Standards

### Docstrings

Use NumPy-style docstrings:

```python
def find_transits_bls(time, flux, min_period=0.5, max_period=100.0):
    """
    Find transits using Box Least Squares algorithm.

    Parameters
    ----------
    time : array-like
        Time array in days.
    flux : array-like
        Normalized flux array.
    min_period : float, optional
        Minimum period to search (days). Default: 0.5
    max_period : float, optional
        Maximum period to search (days). Default: 100.0

    Returns
    -------
    dict
        Dictionary containing:
        - 'period': Best-fit period (days)
        - 'depth': Transit depth
        - 'snr': Signal-to-noise ratio

    Examples
    --------
    >>> result = find_transits_bls(time, flux)
    >>> print(f"Period: {result['period']:.4f} days")

    Notes
    -----
    Uses the astropy BoxLeastSquares implementation.

    References
    ----------
    .. [1] Kovács et al. (2002), A&A, 391, 369
    """
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, Optional, Tuple, Union
import numpy as np

def calculate_snr(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> float:
    ...
```

### Testing

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`
- Aim for >80% code coverage

```python
import pytest
import numpy as np

@pytest.fixture
def synthetic_data():
    """Generate synthetic transit data."""
    time = np.linspace(0, 30, 1000)
    flux = np.ones_like(time)
    return time, flux

def test_bls_detection(synthetic_data):
    """Test BLS correctly detects transit."""
    time, flux = synthetic_data
    result = find_transits_bls(time, flux)
    assert 'period' in result
    assert result['period'] > 0
```

## Project Structure

```
transitkit/
├── src/transitkit/          # Source code
│   ├── __init__.py
│   ├── core.py              # Core analysis functions
│   ├── analysis.py          # Statistical analysis
│   ├── validation.py        # Validation tools
│   ├── visualization.py     # Plotting functions
│   ├── io.py                # Data I/O
│   ├── utils.py             # Utilities
│   ├── nea.py               # NASA Exoplanet Archive
│   ├── cli.py               # Command-line interface
│   └── gui_app.py           # GUI application
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                # Example scripts
├── .github/workflows/       # CI/CD
├── pyproject.toml           # Package configuration
└── README.md
```

## Release Process

1. Update version in `src/transitkit/__init__.py`
2. Update `CHANGELOG.md`
3. Create a GitHub release with tag `vX.Y.Z`
4. CI will automatically publish to PyPI

## Getting Help

- Open an issue for questions
- Join discussions on GitHub Discussions
- Email: arif.solmaz@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
