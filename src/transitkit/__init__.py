"""
TransitKit v2.0: Professional Exoplanet Transit Light Curve Analysis Toolkit
"""
from .__version__ import __version__, __version_tuple__

__author__ = "Arif Solmaz"
__email__ = "arif.solmaz@gmail.com"
__license__ = "MIT"

# Re-export from flat modules (current structure)
from .core import (
    TransitParameters,
    generate_transit_signal_mandel_agol,
    find_transits_bls_advanced,
    find_transits_multiple_methods,
    find_period_gls,
    find_period_pdm,
    add_noise,
)
from .analysis import (
    detrend_light_curve_gp,
    remove_systematics_pca,
    measure_transit_timing_variations,
)
from .utils import (
    calculate_snr,
    estimate_limb_darkening,
    calculate_transit_duration_from_parameters,
)
from .validation import (
    validate_transit_parameters,
    perform_injection_recovery_test,
)
from .io import (
    load_tess_data_advanced,
    export_transit_results,
)
from .visualization import (
    setup_publication_style,
    create_transit_report_figure,
)

__all__ = [
    "__version__", "__version_tuple__",
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
