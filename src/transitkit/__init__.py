"""
TransitKit v2.0: Professional Exoplanet Transit Light Curve Analysis Toolkit
"""

from .__version__ import __version__, __version_tuple__

__author__ = "Arif Solmaz"
__email__ = "arif.solmaz@gmail.com"
__license__ = "MIT"

from .analysis import (
    detrend_light_curve_gp,
    measure_transit_timing_variations,
    remove_systematics_pca,
)

# Re-export from flat modules (current structure)
from .core import (
    TransitParameters,
    add_noise,
    find_period_gls,
    find_period_pdm,
    find_transits_bls_advanced,
    find_transits_multiple_methods,
    generate_transit_signal_mandel_agol,
)
from .io import (
    export_transit_results,
    load_tess_data_advanced,
)
from .utils import (
    calculate_snr,
    calculate_transit_duration_from_parameters,
    estimate_limb_darkening,
)
from .validation import (
    perform_injection_recovery_test,
    validate_transit_parameters,
)
from .visualization import (
    create_transit_report_figure,
    setup_publication_style,
)

__all__ = [
    "__version__",
    "__version_tuple__",
    "TransitParameters",
    "generate_transit_signal_mandel_agol",
    "find_transits_bls_advanced",
    "find_transits_multiple_methods",
    "find_period_gls",
    "find_period_pdm",
    "add_noise",
    "detrend_light_curve_gp",
    "remove_systematics_pca",
    "measure_transit_timing_variations",
    "calculate_snr",
    "validate_transit_parameters",
    "perform_injection_recovery_test",
    "load_tess_data_advanced",
    "export_transit_results",
    "setup_publication_style",
]
