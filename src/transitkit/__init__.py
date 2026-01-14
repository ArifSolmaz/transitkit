"""TransitKit: Exoplanet Transit Light Curve Analysis Toolkit"""

__version__ = "0.1.0"
__author__ = "YOUR NAME"
__email__ = "YOUR-EMAIL@gmail.com"

# Export main functions
from .transit import (
    generate_transit_signal,
    add_noise,
    plot_light_curve,
    find_transits_box,
)

def hello():
    """Simple test function"""
    return f"Hello from TransitKit v{__version__}!"

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'hello',
    'generate_transit_signal',
    'add_noise',
    'plot_light_curve',
    'find_transits_box',
]