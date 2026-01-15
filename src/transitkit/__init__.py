"""
TransitKit v3.0 - Universal Exoplanet Analysis Toolkit
======================================================

The magic: Enter ANY identifier, get COMPLETE analysis.

Supports:
- ANY identifier (TIC, KIC, KOI, TOI, EPIC, HD, planet names, coordinates)
- ALL missions (TESS, Kepler, K2, JWST, ground-based)
- FULL pipeline (download → detection → fitting → publication)

Quick Start:
    >>> from transitkit import UniversalTarget
    >>> target = UniversalTarget("WASP-39 b")
    >>> target.analyze()
    >>> target.export("my_analysis/")

Or work with any identifier:
    >>> target = UniversalTarget("TIC 374829238")
    >>> target = UniversalTarget("TOI-700 d")
    >>> target = UniversalTarget("Kepler-442b")
    >>> target = UniversalTarget("HD 209458 b")
    >>> target = UniversalTarget("307.2312 -47.8731")  # coords

Modules:
    universal   - Target resolution and cross-matching
    missions    - Multi-mission data download
    spectroscopy - JWST transmission/emission spectra
    ml          - ML-powered transit detection
    publication - LaTeX tables and figures
"""

__version__ = "3.0.0"
__author__ = "Arif Solmaz"

# Core imports
from transitkit.universal.resolver import (
    UniversalTarget,
    UniversalResolver,
    CrossMatchedIDs,
    StellarParameters,
    PlanetParameters,
    AvailableData,
    TargetType,
    resolve,
)

from transitkit.missions.downloader import (
    MultiMissionDownloader,
    MultiMissionData,
    LightCurveData,
    download_all,
)

from transitkit.spectroscopy.jwst import (
    JWSTSpectroscopy,
    TransmissionSpectrum,
    MoleculeDetection,
    AtmosphericProperties,
)

from transitkit.ml.detection import (
    MLTransitDetector,
    TransitCandidate,
    DetectionMethod,
    detect_transits,
)

from transitkit.publication.generator import (
    PublicationGenerator,
    PublicationConfig,
)


# Extend UniversalTarget with analysis methods
def _analyze(self, download: bool = True, detect: bool = True, **kwargs):
    """
    Run complete analysis pipeline.

    Args:
        download: Download all available data
        detect: Run transit detection
        **kwargs: Additional arguments

    Returns:
        Dict with analysis results
    """
    results = {
        "target": self,
        "lightcurves": [],
        "candidates": [],
        "transmission_spectrum": None,
        "combined_lc": None,
    }

    # Download data
    if download:
        print("📥 Downloading data...")
        downloader = MultiMissionDownloader(self)
        data = downloader.download_all(verbose=False)
        results["lightcurves"] = data.all_lightcurves
        results["combined_lc"] = data.stitch() if data.all_lightcurves else None

    # Transit detection
    if detect and results["combined_lc"]:
        print("🔍 Detecting transits...")
        detector = MLTransitDetector(self)
        combined = results["combined_lc"]
        candidates = detector.detect(combined.time, combined.flux, combined.flux_err, **kwargs)
        results["candidates"] = candidates

        if candidates:
            print(f"   Found {len(candidates)} candidates")
            for c in candidates[:3]:
                print(
                    f"   • P={c.period:.4f}d, depth={c.depth:.0f}ppm, "
                    f"SNR={c.snr:.1f}, ML={c.ml_score:.2f}"
                )

    # JWST spectroscopy
    if self.available_data.jwst_programs:
        print("🌈 Checking JWST spectroscopy...")
        jwst = JWSTSpectroscopy(self)
        spectrum = jwst.get_transmission_spectrum()
        if spectrum:
            results["transmission_spectrum"] = spectrum
            molecules = jwst.detect_molecules(spectrum)
            results["molecules"] = molecules

            detected = [m for m in molecules if m.detected]
            if detected:
                print(f"   Detected: {', '.join(m.molecule for m in detected[:5])}")

    print("✅ Analysis complete")
    return results


def _export(self, output_dir: str, results: dict = None, **kwargs):
    """
    Export analysis results for publication.

    Args:
        output_dir: Output directory
        results: Analysis results (runs analyze() if None)
        **kwargs: Additional arguments
    """
    if results is None:
        results = self.analyze(**kwargs)

    pub = PublicationGenerator(self, results)
    pub.generate_all(output_dir)


def _get_lightcurves(self, missions: list = None, **kwargs):
    """
    Get all light curves for this target.

    Args:
        missions: Filter by mission names
        **kwargs: Passed to downloader

    Returns:
        List of LightCurveData
    """
    downloader = MultiMissionDownloader(self)
    data = downloader.download_all(**kwargs)

    lcs = data.all_lightcurves

    if missions:
        lcs = [lc for lc in lcs if lc.mission in missions]

    return lcs


def _get_transmission_spectrum(self, **kwargs):
    """
    Get JWST transmission spectrum.

    Returns:
        TransmissionSpectrum or None
    """
    jwst = JWSTSpectroscopy(self)
    return jwst.get_transmission_spectrum(**kwargs)


# Attach methods to UniversalTarget
UniversalTarget.analyze = _analyze
UniversalTarget.export = _export
UniversalTarget.get_lightcurves = _get_lightcurves
UniversalTarget.get_transmission_spectrum = _get_transmission_spectrum


# Convenience functions
def analyze(identifier: str, **kwargs) -> dict:
    """
    Quick analysis of any target.

    Args:
        identifier: Any planet/star identifier
        **kwargs: Passed to analyze()

    Returns:
        Analysis results dict
    """
    target = UniversalTarget(identifier)
    return target.analyze(**kwargs)


def quick_look(identifier: str):
    """
    Quick look at a target - just resolve and show info.

    Args:
        identifier: Any planet/star identifier
    """
    target = UniversalTarget(identifier, verbose=True)
    return target


# What gets exported with `from transitkit import *`
__all__ = [
    # Main class
    "UniversalTarget",
    # Resolvers
    "UniversalResolver",
    "CrossMatchedIDs",
    "StellarParameters",
    "PlanetParameters",
    "AvailableData",
    "TargetType",
    "resolve",
    # Data
    "MultiMissionDownloader",
    "MultiMissionData",
    "LightCurveData",
    "download_all",
    # Spectroscopy
    "JWSTSpectroscopy",
    "TransmissionSpectrum",
    "MoleculeDetection",
    "AtmosphericProperties",
    # Detection
    "MLTransitDetector",
    "TransitCandidate",
    "DetectionMethod",
    "detect_transits",
    # Publication
    "PublicationGenerator",
    "PublicationConfig",
    # Convenience
    "analyze",
    "quick_look",
]
