import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys
from typing import Generator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def synthetic_transit_light_curve():
    """Create synthetic transit light curve for testing."""
    np.random.seed(42)
    
    # Time array
    time = np.linspace(0, 30, 3000)
    
    # Transit parameters
    params = {
        't0': 15.0,
        'period': 5.0,
        'rp_over_rs': 0.1,
        'a_over_rs': 15.0,
        'inc': 89.0,
        'u1': 0.5,
        'u2': 0.0
    }
    
    # Create transit model
    from transitkit.core.models import TransitModelJAX
    model = TransitModelJAX()
    flux = model.compute(time, params)
    
    # Add noise
    flux += np.random.normal(0, 0.001, len(time))
    flux_err = np.ones_like(flux) * 0.001
    
    # Create light curve object
    from transitkit.data.pipeline import LightCurveData
    lc = LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=np.zeros_like(flux),
        cadence=time[1] - time[0],
        mission='TEST',
        target_id='TEST-001'
    )
    
    return lc

@pytest.fixture
def sample_fit_result(synthetic_transit_light_curve):
    """Create sample fit result for testing."""
    from transitkit.fitting.fitters import FittingOrchestrator
    
    orchestrator = FittingOrchestrator()
    result = orchestrator.fit_transit(
        synthetic_transit_light_curve,
        method='least_squares'
    )
    
    return result

@pytest.fixture(scope="session")
def example_light_curve_file(test_data_dir) -> Path:
    """Path to example light curve file."""
    file_path = test_data_dir / "example_light_curve.csv"
    
    # Create example data if it doesn't exist
    if not file_path.exists():
        import pandas as pd
        
        time = np.linspace(0, 10, 1000)
        flux = 1 + 0.01 * np.sin(2 * np.pi * time / 5)
        flux_err = np.ones_like(flux) * 0.001
        
        df = pd.DataFrame({
            'time': time,
            'flux': flux,
            'flux_err': flux_err
        })
        df.to_csv(file_path, index=False)
    
    return file_path

# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: integration test"
    )
    config.addinivalue_line(
        "markers",
        "gpu: requires GPU"
    )
    config.addinivalue_line(
        "markers",
        "web: web interface tests"
    )