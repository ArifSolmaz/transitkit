"""
TransitKit v3.0 - Multi-Mission Data Downloader

Automatically download ALL available data for any resolved target:
- TESS (all sectors, all cadences)
- Kepler (all quarters)
- K2 (all campaigns)
- JWST (spectroscopy + imaging)
- Ground-based (ExoFOP)

Example:
    >>> target = UniversalTarget("WASP-39 b")
    >>> downloader = MultiMissionDownloader(target)
    >>> all_data = downloader.download_all()
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# Lazy imports
def _import_lightkurve():
    import lightkurve as lk

    return lk


def _import_astroquery():
    from astroquery.mast import Observations, Catalogs

    return Observations, Catalogs


def _import_astropy():
    from astropy.io import fits
    from astropy.table import Table
    from astropy import units as u
    from astropy.time import Time

    return fits, Table, u, Time


@dataclass
class LightCurveData:
    """Container for light curve data from any mission."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mission: str
    sector: Optional[int] = None  # TESS
    quarter: Optional[int] = None  # Kepler
    campaign: Optional[int] = None  # K2
    cadence: Optional[str] = None  # '20s', '2min', 'long'
    author: Optional[str] = None  # SPOC, QLP, etc.
    quality_mask: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Generate descriptive label."""
        parts = [self.mission]
        if self.sector:
            parts.append(f"S{self.sector}")
        if self.quarter:
            parts.append(f"Q{self.quarter}")
        if self.campaign:
            parts.append(f"C{self.campaign}")
        if self.cadence:
            parts.append(self.cadence)
        if self.author:
            parts.append(self.author)
        return "_".join(parts)

    def to_lightkurve(self):
        """Convert to lightkurve LightCurve object."""
        lk = _import_lightkurve()
        return lk.LightCurve(
            time=self.time, flux=self.flux, flux_err=self.flux_err, label=self.label
        )


@dataclass
class SpectrumData:
    """Container for spectroscopic data (JWST, etc.)."""

    wavelength: np.ndarray  # microns
    flux: np.ndarray  # Various units
    flux_err: np.ndarray
    instrument: str  # NIRSpec, MIRI, etc.
    mode: str  # transit, eclipse, phase
    program_id: str
    visit: Optional[int] = None
    integration: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"JWST_{self.instrument}_{self.program_id}"


@dataclass
class MultiMissionData:
    """Container for all data from a target."""

    target_name: str
    tic: Optional[int] = None
    kic: Optional[int] = None

    # Light curves by mission
    tess: List[LightCurveData] = field(default_factory=list)
    kepler: List[LightCurveData] = field(default_factory=list)
    k2: List[LightCurveData] = field(default_factory=list)
    ground: List[LightCurveData] = field(default_factory=list)

    # Spectra
    jwst_spectra: List[SpectrumData] = field(default_factory=list)

    @property
    def all_lightcurves(self) -> List[LightCurveData]:
        """All light curves combined."""
        return self.tess + self.kepler + self.k2 + self.ground

    @property
    def total_timespan(self) -> float:
        """Total time baseline in days."""
        all_times = []
        for lc in self.all_lightcurves:
            all_times.extend(lc.time)
        if not all_times:
            return 0.0
        return max(all_times) - min(all_times)

    def stitch(self, normalize: bool = True) -> LightCurveData:
        """
        Stitch all light curves into one combined dataset.

        Args:
            normalize: Normalize each segment to median flux
        """
        all_time = []
        all_flux = []
        all_err = []

        for lc in self.all_lightcurves:
            t, f, e = lc.time, lc.flux, lc.flux_err

            if normalize:
                median = np.nanmedian(f)
                f = f / median
                e = e / median

            all_time.extend(t)
            all_flux.extend(f)
            all_err.extend(e)

        # Sort by time
        sort_idx = np.argsort(all_time)

        return LightCurveData(
            time=np.array(all_time)[sort_idx],
            flux=np.array(all_flux)[sort_idx],
            flux_err=np.array(all_err)[sort_idx],
            mission="Combined",
            metadata={"sources": [lc.label for lc in self.all_lightcurves]},
        )

    def summary(self) -> str:
        """Text summary of available data."""
        lines = [
            f"MultiMissionData: {self.target_name}",
            f"  TESS: {len(self.tess)} light curves",
            f"  Kepler: {len(self.kepler)} light curves",
            f"  K2: {len(self.k2)} light curves",
            f"  Ground: {len(self.ground)} light curves",
            f"  JWST spectra: {len(self.jwst_spectra)}",
            f"  Total timespan: {self.total_timespan:.1f} days",
        ]
        return "\n".join(lines)


class MultiMissionDownloader:
    """
    Download all available data for a UniversalTarget.

    Example:
        >>> target = UniversalTarget("WASP-39 b")
        >>> downloader = MultiMissionDownloader(target)
        >>> data = downloader.download_all()
        >>> print(data.summary())
    """

    def __init__(self, target, cache_dir: Optional[str] = None):
        """
        Initialize downloader for a target.

        Args:
            target: UniversalTarget instance
            cache_dir: Directory to cache downloaded data
        """
        self.target = target
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".transitkit_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self._data = MultiMissionData(
            target_name=target.ids.planet_name or target.identifier, tic=target.tic, kic=target.kic
        )

    def download_all(
        self,
        tess: bool = True,
        kepler: bool = True,
        k2: bool = True,
        jwst: bool = True,
        parallel: bool = True,
        verbose: bool = True,
    ) -> MultiMissionData:
        """
        Download all available data.

        Args:
            tess: Download TESS data
            kepler: Download Kepler data
            k2: Download K2 data
            jwst: Download JWST data
            parallel: Use parallel downloads
            verbose: Print progress

        Returns:
            MultiMissionData container
        """
        if verbose:
            print(f"📥 Downloading all data for {self.target.identifier}")

        tasks = []

        if tess and self.target.available_data.tess_sectors:
            if verbose:
                print(f"   TESS: {len(self.target.available_data.tess_sectors)} sectors")
            tasks.append(("TESS", self._download_tess))

        if kepler and self.target.available_data.kepler_quarters:
            if verbose:
                print(f"   Kepler: {len(self.target.available_data.kepler_quarters)} quarters")
            tasks.append(("Kepler", self._download_kepler))

        if k2 and self.target.available_data.k2_campaigns:
            if verbose:
                print(f"   K2: {len(self.target.available_data.k2_campaigns)} campaigns")
            tasks.append(("K2", self._download_k2))

        if jwst and self.target.available_data.jwst_programs:
            if verbose:
                print(f"   JWST: {len(self.target.available_data.jwst_programs)} programs")
            tasks.append(("JWST", self._download_jwst))

        if parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(func): name for name, func in tasks}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                        if verbose:
                            print(f"   ✓ {name} complete")
                    except Exception as e:
                        if verbose:
                            print(f"   ✗ {name} failed: {e}")
        else:
            for name, func in tasks:
                try:
                    func()
                    if verbose:
                        print(f"   ✓ {name} complete")
                except Exception as e:
                    if verbose:
                        print(f"   ✗ {name} failed: {e}")

        if verbose:
            print(f"\n{self._data.summary()}")

        return self._data

    def _download_tess(self):
        """Download all available TESS data."""
        lk = _import_lightkurve()

        tic = self.target.tic
        if not tic:
            warnings.warn("No TIC ID available for TESS download")
            return

        # Search for all TESS data
        search_result = lk.search_lightcurve(f"TIC {tic}", mission="TESS")

        if search_result is None or len(search_result) == 0:
            return

        # Download all
        for i, row in enumerate(search_result):
            try:
                lc = row.download()
                if lc is None:
                    continue

                # Extract metadata
                sector = getattr(lc.meta, "SECTOR", None)
                author = getattr(
                    lc.meta, "AUTHOR", str(row.author) if hasattr(row, "author") else None
                )
                exptime = getattr(lc.meta, "TIMEDEL", None)

                # Determine cadence
                if exptime:
                    exptime_sec = exptime * 86400  # days to seconds
                    if exptime_sec < 30:
                        cadence = "20s"
                    elif exptime_sec < 200:
                        cadence = "2min"
                    else:
                        cadence = "FFI"
                else:
                    cadence = None

                lc_data = LightCurveData(
                    time=np.array(lc.time.value),
                    flux=np.array(lc.flux.value),
                    flux_err=(
                        np.array(lc.flux_err.value)
                        if lc.flux_err is not None
                        else np.zeros_like(lc.flux.value)
                    ),
                    mission="TESS",
                    sector=sector,
                    cadence=cadence,
                    author=author,
                    quality_mask=np.array(lc.quality) if hasattr(lc, "quality") else None,
                    metadata=dict(lc.meta) if lc.meta else {},
                )

                self._data.tess.append(lc_data)

            except Exception as e:
                warnings.warn(f"Failed to download TESS data {i}: {e}")

    def _download_kepler(self):
        """Download all available Kepler data."""
        lk = _import_lightkurve()

        kic = self.target.kic
        if not kic:
            # Try coordinate search
            if self.target.stellar.ra and self.target.stellar.dec:
                search_result = lk.search_lightcurve(
                    f"{self.target.stellar.ra} {self.target.stellar.dec}", mission="Kepler"
                )
            else:
                warnings.warn("No KIC ID or coordinates available for Kepler download")
                return
        else:
            search_result = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")

        if search_result is None or len(search_result) == 0:
            return

        for i, row in enumerate(search_result):
            try:
                lc = row.download()
                if lc is None:
                    continue

                quarter = getattr(lc.meta, "QUARTER", None)

                lc_data = LightCurveData(
                    time=np.array(lc.time.value),
                    flux=np.array(lc.flux.value),
                    flux_err=(
                        np.array(lc.flux_err.value)
                        if lc.flux_err is not None
                        else np.zeros_like(lc.flux.value)
                    ),
                    mission="Kepler",
                    quarter=quarter,
                    cadence=(
                        "long" if getattr(lc.meta, "OBSMODE", "") == "long cadence" else "short"
                    ),
                    metadata=dict(lc.meta) if lc.meta else {},
                )

                self._data.kepler.append(lc_data)

            except Exception as e:
                warnings.warn(f"Failed to download Kepler data {i}: {e}")

    def _download_k2(self):
        """Download all available K2 data."""
        lk = _import_lightkurve()

        epic = self.target.ids.epic

        if epic:
            search_result = lk.search_lightcurve(f"EPIC {epic}", mission="K2")
        elif self.target.stellar.ra and self.target.stellar.dec:
            search_result = lk.search_lightcurve(
                f"{self.target.stellar.ra} {self.target.stellar.dec}", mission="K2"
            )
        else:
            return

        if search_result is None or len(search_result) == 0:
            return

        for i, row in enumerate(search_result):
            try:
                lc = row.download()
                if lc is None:
                    continue

                campaign = getattr(lc.meta, "CAMPAIGN", None)

                lc_data = LightCurveData(
                    time=np.array(lc.time.value),
                    flux=np.array(lc.flux.value),
                    flux_err=(
                        np.array(lc.flux_err.value)
                        if lc.flux_err is not None
                        else np.zeros_like(lc.flux.value)
                    ),
                    mission="K2",
                    campaign=campaign,
                    metadata=dict(lc.meta) if lc.meta else {},
                )

                self._data.k2.append(lc_data)

            except Exception as e:
                warnings.warn(f"Failed to download K2 data {i}: {e}")

    def _download_jwst(self):
        """Download available JWST spectroscopic data."""
        Observations, _ = _import_astroquery()
        fits, Table, u, _ = _import_astropy()

        if not self.target.stellar.ra or not self.target.stellar.dec:
            return

        try:
            from astropy.coordinates import SkyCoord

            coord = SkyCoord(ra=self.target.stellar.ra, dec=self.target.stellar.dec, unit="deg")

            # Query JWST observations
            obs_table = Observations.query_region(coord, radius=0.01 * u.deg)

            if obs_table is None or len(obs_table) == 0:
                return

            # Filter for JWST spectroscopy
            jwst_mask = (obs_table["obs_collection"] == "JWST") & (
                obs_table["dataproduct_type"]
                .astype(str)
                .str.contains("spectrum", case=False, na=False)
            )

            jwst_obs = obs_table[jwst_mask]

            for row in jwst_obs:
                try:
                    # Get data products
                    products = Observations.get_product_list(row)

                    # Filter for x1d (extracted 1D spectra)
                    x1d_products = Observations.filter_products(
                        products, productSubGroupDescription="X1D"
                    )

                    if len(x1d_products) > 0:
                        # Download
                        manifest = Observations.download_products(
                            x1d_products, download_dir=str(self.cache_dir / "jwst")
                        )

                        # Parse downloaded files
                        for local_path in manifest["Local Path"]:
                            if local_path.endswith(".fits"):
                                self._parse_jwst_spectrum(local_path, row)

                except Exception as e:
                    warnings.warn(f"Failed to download JWST product: {e}")

        except Exception as e:
            warnings.warn(f"JWST download failed: {e}")

    def _parse_jwst_spectrum(self, filepath: str, obs_row):
        """Parse a JWST x1d spectrum file."""
        fits, _, _, _ = _import_astropy()

        try:
            with fits.open(filepath) as hdul:
                # Find spectrum extension
                for ext in hdul:
                    if ext.name == "EXTRACT1D" or (hasattr(ext, "data") and ext.data is not None):
                        if hasattr(ext.data, "dtype") and "WAVELENGTH" in ext.data.dtype.names:
                            spectrum = SpectrumData(
                                wavelength=ext.data["WAVELENGTH"],
                                flux=ext.data["FLUX"],
                                flux_err=(
                                    ext.data["FLUX_ERROR"]
                                    if "FLUX_ERROR" in ext.data.dtype.names
                                    else np.zeros_like(ext.data["FLUX"])
                                ),
                                instrument=str(obs_row.get("instrument_name", "UNKNOWN")),
                                mode="transit",  # Would need metadata to determine
                                program_id=str(obs_row.get("proposal_id", "")),
                                metadata={"filename": filepath},
                            )
                            self._data.jwst_spectra.append(spectrum)
                            break

        except Exception as e:
            warnings.warn(f"Failed to parse JWST spectrum {filepath}: {e}")

    def download_tess(
        self, sectors: Optional[List[int]] = None, cadence: Optional[str] = None
    ) -> List[LightCurveData]:
        """
        Download specific TESS data.

        Args:
            sectors: List of sectors to download (None = all)
            cadence: Cadence filter ('20s', '2min', 'FFI')
        """
        self._download_tess()

        result = self._data.tess

        if sectors:
            result = [lc for lc in result if lc.sector in sectors]
        if cadence:
            result = [lc for lc in result if lc.cadence == cadence]

        return result

    def download_kepler(self, quarters: Optional[List[int]] = None) -> List[LightCurveData]:
        """Download specific Kepler data."""
        self._download_kepler()

        result = self._data.kepler

        if quarters:
            result = [lc for lc in result if lc.quarter in quarters]

        return result


def download_all(target, **kwargs) -> MultiMissionData:
    """
    Convenience function to download all data for a target.

    Args:
        target: UniversalTarget instance
        **kwargs: Passed to MultiMissionDownloader.download_all()

    Returns:
        MultiMissionData container
    """
    downloader = MultiMissionDownloader(target)
    return downloader.download_all(**kwargs)
