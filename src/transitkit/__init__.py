"""TransitKit v2.0: Professional Exoplanet Transit Light Curve Analysis Toolkit"""

__version__ = "2.0.0"
__author__ = "Arif Solmaz"
__email__ = "arif.solmaz@gmail.com"
__license__ = "MIT"
__citation__ = "TransitKit v2.0 (2024) - For research use"

def hello():
    """Simple test function"""
    return f"Hello from TransitKit v{__version__}!"

# Lazy exports for backward compatibility
def generate_transit_signal(*args, **kwargs):
    """Backward compatibility wrapper. Use core.generate_transit_signal_advanced for new code."""
    from .core import generate_transit_signal_mandel_agol
    return generate_transit_signal_mandel_agol(*args, **kwargs)

def add_noise(*args, **kwargs):
    """Backward compatibility wrapper."""
    from .core import add_noise as _f
    return _f(*args, **kwargs)

def plot_light_curve(*args, **kwargs):
    """Backward compatibility wrapper. Use visualization.plot_transit_summary for publication plots."""
    from .visualization import plot_light_curve_basic
    return plot_light_curve_basic(*args, **kwargs)

def find_transits_box(*args, **kwargs):
    """Backward compatibility wrapper. Use core.find_transits_bls_advanced for scientific analysis."""
    from .core import find_transits_bls_advanced
    return find_transits_bls_advanced(*args, **kwargs)

# Lazy imports for new modules (prevents import-time failures)
def _import_core():
    """Lazy import core module."""
    try:
        from . import core
        return core
    except ImportError as e:
        raise ImportError(f"Failed to import core module: {e}")

def _import_analysis():
    """Lazy import analysis module."""
    try:
        from . import analysis
        return analysis
    except ImportError as e:
        raise ImportError(f"Failed to import analysis module: {e}")

def _import_visualization():
    """Lazy import visualization module."""
    try:
        from . import visualization
        return visualization
    except ImportError as e:
        raise ImportError(f"Failed to import visualization module: {e}")

def _import_io():
    """Lazy import io module."""
    try:
        from . import io
        return io
    except ImportError as e:
        raise ImportError(f"Failed to import io module: {e}")

def _import_utils():
    """Lazy import utils module."""
    try:
        from . import utils
        return utils
    except ImportError as e:
        raise ImportError(f"Failed to import utils module: {e}")

def _import_validation():
    """Lazy import validation module."""
    try:
        from . import validation
        return validation
    except ImportError as e:
        raise ImportError(f"Failed to import validation module: {e}")

def _import_nea():
    """Lazy import nea module."""
    try:
        from . import nea
        return nea
    except ImportError as e:
        raise ImportError(f"Failed to import nea module: {e}")

# Property-based lazy access to modules
class _ModuleProxy:
    """Proxy for lazy module loading."""
    def __init__(self, import_func):
        self._import_func = import_func
        self._module = None
    
    @property
    def module(self):
        if self._module is None:
            self._module = self._import_func()
        return self._module
    
    def __getattr__(self, name):
        return getattr(self.module, name)

# Create module proxies
core = _ModuleProxy(_import_core)
analysis = _ModuleProxy(_import_analysis)
visualization = _ModuleProxy(_import_visualization)
io = _ModuleProxy(_import_io)
utils = _ModuleProxy(_import_utils)
validation = _ModuleProxy(_import_validation)
nea = _ModuleProxy(_import_nea)

# Convenience function exports (direct access to commonly used functions)
def lookup_planet(*args, **kwargs):
    """NASA Exoplanet Archive lookup (from nea module)."""
    return nea.lookup_planet(*args, **kwargs)

def load_tess_data(*args, **kwargs):
    """Load TESS data (from io module)."""
    return io.load_tess_data_advanced(*args, **kwargs)

def generate_transit_signal_advanced(*args, **kwargs):
    """Advanced transit generation with Mandel & Agol model."""
    return core.generate_transit_signal_mandel_agol(*args, **kwargs)

def find_transits_multiple(*args, **kwargs):
    """Find transits using multiple methods for robust detection."""
    return core.find_transits_multiple_methods(*args, **kwargs)

def detrend_gp(*args, **kwargs):
    """Detrend light curve using Gaussian Process."""
    return analysis.detrend_light_curve_gp(*args, **kwargs)

def create_transit_report(*args, **kwargs):
    """Create publication-quality transit report figure."""
    return visualization.create_transit_report_figure(*args, **kwargs)

def estimate_parameters_mcmc(*args, **kwargs):
    """Estimate transit parameters using MCMC."""
    return core.estimate_parameters_mcmc(*args, **kwargs)

def validate_transit_detection(*args, **kwargs):
    """Validate transit detection with multiple tests."""
    return validation.validate_transit_detection(*args, **kwargs)

def measure_ttvs(*args, **kwargs):
    """Measure Transit Timing Variations."""
    return analysis.measure_transit_timing_variations(*args, **kwargs)

# Export everything
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__citation__",
    
    # Core functions (backward compatible)
    "hello",
    "generate_transit_signal",
    "add_noise",
    "plot_light_curve",
    "find_transits_box",
    "lookup_planet",
    
    # Module access
    "core",
    "analysis",
    "visualization",
    "io",
    "utils",
    "validation",
    "nea",
    
    # Convenience functions
    "load_tess_data",
    "generate_transit_signal_advanced",
    "find_transits_multiple",
    "detrend_gp",
    "create_transit_report",
    "estimate_parameters_mcmc",
    "validate_transit_detection",
    "measure_ttvs",
]

# Add deprecated warnings for old functions
import warnings
import functools

def _deprecated(message):
    """Decorator for deprecated functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__}: {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Mark old functions as deprecated (but still functional)
generate_transit_signal = _deprecated(
    "Use generate_transit_signal_advanced() for Mandel & Agol models"
)(generate_transit_signal)

find_transits_box = _deprecated(
    "Use find_transits_multiple() for robust detection or core.find_transits_bls_advanced()"
)(find_transits_box)

plot_light_curve = _deprecated(
    "Use visualization.plot_transit_summary() for publication-quality plots"
)(plot_light_curve)

# Version check for dependencies
def _check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    required = [
        ("numpy", "1.21"),
        ("matplotlib", "3.5"),
        ("scipy", "1.8"),
        ("astropy", "5.1"),
    ]
    
    for pkg, min_version in required:
        try:
            import importlib.metadata
            version = importlib.metadata.version(pkg)
            from packaging import version as packaging_version
            if packaging_version.parse(version) < packaging_version.parse(min_version):
                missing.append(f"{pkg}>={min_version} (found {version})")
        except ImportError:
            missing.append(f"{pkg}>={min_version}")
    
    if missing:
        warnings.warn(
            f"TransitKit v2.0 may require updated dependencies. Missing/old: {', '.join(missing)}",
            UserWarning,
            stacklevel=2
        )

# Run dependency check on import
_check_dependencies()