"""TransitKit: Exoplanet Transit Light Curve Analysis Toolkit"""

__version__ = "0.1.0"
__author__ = "Arif Solmaz"
__email__ = "arif.solmaz@gmail.com"

def hello():
    """Simple test function"""
    return f"Hello from TransitKit v{__version__}!"

# Lazy exports (avoids import-time failures)
def generate_transit_signal(*args, **kwargs):
    from .transit import generate_transit_signal as _f
    return _f(*args, **kwargs)

def add_noise(*args, **kwargs):
    from .transit import add_noise as _f
    return _f(*args, **kwargs)

def plot_light_curve(*args, **kwargs):
    from .transit import plot_light_curve as _f
    return _f(*args, **kwargs)

def find_transits_box(*args, **kwargs):
    from .transit import find_transits_box as _f
    return _f(*args, **kwargs)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "hello",
    "generate_transit_signal",
    "add_noise",
    "plot_light_curve",
    "find_transits_box",
]
