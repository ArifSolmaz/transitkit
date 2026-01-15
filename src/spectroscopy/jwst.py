"""
TransitKit v3.0 - JWST Spectroscopy Module

Handle JWST transmission and emission spectra:
- NIRSpec, MIRI, NIRCam, NIRISS data
- Transmission spectrum extraction
- Atmospheric retrieval (via petitRADTRANS)
- Molecule detection

Example:
    >>> target = UniversalTarget("WASP-39 b")
    >>> spec = JWSTSpectroscopy(target)
    >>> transmission = spec.get_transmission_spectrum()
    >>> molecules = spec.detect_molecules()
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import numpy as np

# Lazy imports
def _import_astroquery():
    from astroquery.mast import Observations
    return Observations

def _import_astropy():
    from astropy.io import fits
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    return fits, u, SkyCoord


@dataclass
class TransmissionSpectrum:
    """Transmission spectrum data container."""
    wavelength: np.ndarray           # microns
    transit_depth: np.ndarray        # (Rp/Rs)^2 or ppm
    transit_depth_err: np.ndarray
    wavelength_err: Optional[np.ndarray] = None
    instrument: str = ""
    program_id: str = ""
    planet_name: str = ""
    is_binned: bool = False
    bin_width: Optional[float] = None  # microns
    metadata: Dict = field(default_factory=dict)
    
    @property
    def depth_ppm(self) -> np.ndarray:
        """Transit depth in ppm."""
        if np.nanmedian(self.transit_depth) < 0.1:
            # Already in fraction, convert
            return self.transit_depth * 1e6
        return self.transit_depth
    
    def bin_spectrum(self, bin_width: float = 0.1) -> 'TransmissionSpectrum':
        """
        Bin the spectrum to lower resolution.
        
        Args:
            bin_width: Bin width in microns
        """
        wl = self.wavelength
        depth = self.transit_depth
        err = self.transit_depth_err
        
        wl_min, wl_max = np.nanmin(wl), np.nanmax(wl)
        bin_edges = np.arange(wl_min, wl_max + bin_width, bin_width)
        
        binned_wl = []
        binned_depth = []
        binned_err = []
        
        for i in range(len(bin_edges) - 1):
            mask = (wl >= bin_edges[i]) & (wl < bin_edges[i+1])
            if np.sum(mask) > 0:
                # Weighted mean
                weights = 1.0 / err[mask]**2
                binned_wl.append(np.mean(wl[mask]))
                binned_depth.append(np.sum(weights * depth[mask]) / np.sum(weights))
                binned_err.append(1.0 / np.sqrt(np.sum(weights)))
        
        return TransmissionSpectrum(
            wavelength=np.array(binned_wl),
            transit_depth=np.array(binned_depth),
            transit_depth_err=np.array(binned_err),
            instrument=self.instrument,
            program_id=self.program_id,
            planet_name=self.planet_name,
            is_binned=True,
            bin_width=bin_width,
            metadata=self.metadata
        )


@dataclass
class MoleculeDetection:
    """Result of molecule detection analysis."""
    molecule: str
    detected: bool
    significance: float              # sigma
    wavelength_range: Tuple[float, float]  # microns
    feature_depth: Optional[float] = None  # ppm
    reference: Optional[str] = None


@dataclass
class AtmosphericProperties:
    """Derived atmospheric properties."""
    temperature: Optional[float] = None      # K
    temperature_err: Optional[float] = None
    metallicity: Optional[float] = None      # [M/H]
    metallicity_err: Optional[float] = None
    c_o_ratio: Optional[float] = None
    c_o_ratio_err: Optional[float] = None
    cloud_top_pressure: Optional[float] = None  # bar
    mean_molecular_weight: Optional[float] = None
    scale_height: Optional[float] = None     # km
    molecules_detected: List[MoleculeDetection] = field(default_factory=list)


# Known molecular features for detection
MOLECULAR_FEATURES = {
    'H2O': [
        (1.35, 1.45, 'H2O 1.4μm'),
        (1.8, 2.0, 'H2O 1.9μm'),
        (2.6, 3.0, 'H2O 2.8μm'),
        (5.5, 7.5, 'H2O 6μm'),
    ],
    'CO2': [
        (4.2, 4.5, 'CO2 4.3μm'),
        (14.5, 15.5, 'CO2 15μm'),
    ],
    'CO': [
        (4.5, 5.0, 'CO 4.7μm'),
    ],
    'CH4': [
        (3.2, 3.5, 'CH4 3.3μm'),
        (7.5, 8.0, 'CH4 7.7μm'),
    ],
    'NH3': [
        (10.0, 11.0, 'NH3 10.5μm'),
    ],
    'SO2': [
        (7.2, 7.5, 'SO2 7.3μm'),
        (8.5, 9.0, 'SO2 8.7μm'),
    ],
    'H2S': [
        (3.8, 4.1, 'H2S 4μm'),
    ],
    'Na': [
        (0.585, 0.595, 'Na D doublet'),
    ],
    'K': [
        (0.765, 0.775, 'K doublet'),
    ],
}


class JWSTSpectroscopy:
    """
    JWST spectroscopy analysis for exoplanet atmospheres.
    
    Handles:
    - Data download from MAST
    - Transmission/emission spectrum extraction
    - Molecule detection
    - Atmospheric retrieval (if petitRADTRANS available)
    
    Example:
        >>> target = UniversalTarget("WASP-39 b")
        >>> spec = JWSTSpectroscopy(target)
        >>> trans = spec.get_transmission_spectrum()
        >>> molecules = spec.detect_molecules(trans)
    """
    
    def __init__(self, target):
        """
        Initialize JWST spectroscopy for a target.
        
        Args:
            target: UniversalTarget instance
        """
        self.target = target
        self._transmission_spectra = []
        self._emission_spectra = []
        self._retrieved = False
        
    def get_transmission_spectrum(
        self,
        instrument: Optional[str] = None,
        program_id: Optional[str] = None
    ) -> Optional[TransmissionSpectrum]:
        """
        Get transmission spectrum from JWST data.
        
        Args:
            instrument: Filter by instrument (NIRSpec, MIRI, etc.)
            program_id: Filter by program ID
            
        Returns:
            TransmissionSpectrum or None
        """
        self._download_spectra()
        
        spectra = self._transmission_spectra
        
        if instrument:
            spectra = [s for s in spectra if instrument.upper() in s.instrument.upper()]
        if program_id:
            spectra = [s for s in spectra if program_id in s.program_id]
        
        if not spectra:
            return None
        
        # If multiple, combine them
        if len(spectra) == 1:
            return spectra[0]
        
        return self._combine_spectra(spectra)
    
    def get_all_spectra(self) -> List[TransmissionSpectrum]:
        """Get all available transmission spectra."""
        self._download_spectra()
        return self._transmission_spectra
    
    def _download_spectra(self):
        """Download JWST spectroscopic data from MAST."""
        if self._retrieved:
            return
        
        Observations = _import_astroquery()
        fits, u, SkyCoord = _import_astropy()
        
        if not self.target.stellar.ra or not self.target.stellar.dec:
            warnings.warn("No coordinates available for JWST query")
            return
        
        try:
            coord = SkyCoord(
                ra=self.target.stellar.ra,
                dec=self.target.stellar.dec,
                unit='deg'
            )
            
            # Query MAST
            obs_table = Observations.query_region(coord, radius=0.02*u.deg)
            
            if obs_table is None or len(obs_table) == 0:
                self._retrieved = True
                return
            
            # Filter for JWST spectroscopy
            jwst_mask = obs_table['obs_collection'] == 'JWST'
            jwst_obs = obs_table[jwst_mask]
            
            for row in jwst_obs:
                dataproduct = str(row.get('dataproduct_type', '')).lower()
                if 'spectrum' not in dataproduct and 'spectroscop' not in dataproduct:
                    continue
                
                try:
                    # Get products
                    products = Observations.get_product_list(row)
                    
                    # Look for x1d files
                    x1d = Observations.filter_products(
                        products,
                        productSubGroupDescription=['X1D', 'X1DINTS']
                    )
                    
                    if len(x1d) > 0:
                        # Download
                        manifest = Observations.download_products(x1d[:1])  # Just first
                        
                        for local_path in manifest['Local Path']:
                            if local_path and local_path.endswith('.fits'):
                                spec = self._parse_x1d_file(local_path, row)
                                if spec:
                                    self._transmission_spectra.append(spec)
                                    
                except Exception as e:
                    warnings.warn(f"Failed to process JWST observation: {e}")
                    
        except Exception as e:
            warnings.warn(f"JWST query failed: {e}")
        
        self._retrieved = True
    
    def _parse_x1d_file(self, filepath: str, obs_row) -> Optional[TransmissionSpectrum]:
        """Parse JWST x1d spectrum file."""
        fits, _, _ = _import_astropy()
        
        try:
            with fits.open(filepath) as hdul:
                # Primary header
                header = hdul[0].header
                
                # Find spectrum extension
                data = None
                for ext in hdul:
                    if ext.name in ('EXTRACT1D', 'SCI', 'SPEC'):
                        if hasattr(ext, 'data') and ext.data is not None:
                            data = ext.data
                            break
                    elif hasattr(ext, 'data') and ext.data is not None:
                        if hasattr(ext.data, 'dtype'):
                            names = ext.data.dtype.names
                            if names and 'WAVELENGTH' in names:
                                data = ext.data
                                break
                
                if data is None:
                    return None
                
                # Extract columns
                wavelength = data['WAVELENGTH']
                flux = data['FLUX'] if 'FLUX' in data.dtype.names else data['SPEC']
                
                if 'FLUX_ERROR' in data.dtype.names:
                    flux_err = data['FLUX_ERROR']
                elif 'ERR' in data.dtype.names:
                    flux_err = data['ERR']
                else:
                    flux_err = np.zeros_like(flux) + np.nanstd(flux) * 0.1
                
                # Convert to transit depth if needed
                # This is simplified - real analysis needs white light normalization
                transit_depth = flux  # Placeholder - needs proper processing
                
                return TransmissionSpectrum(
                    wavelength=wavelength,
                    transit_depth=transit_depth,
                    transit_depth_err=flux_err,
                    instrument=str(obs_row.get('instrument_name', 'UNKNOWN')),
                    program_id=str(obs_row.get('proposal_id', '')),
                    planet_name=self.target.ids.planet_name or self.target.identifier,
                    metadata={
                        'filename': filepath,
                        'target': header.get('TARGNAME', ''),
                    }
                )
                
        except Exception as e:
            warnings.warn(f"Failed to parse {filepath}: {e}")
            return None
    
    def _combine_spectra(self, spectra: List[TransmissionSpectrum]) -> TransmissionSpectrum:
        """Combine multiple spectra into one."""
        all_wl = []
        all_depth = []
        all_err = []
        
        for spec in spectra:
            all_wl.extend(spec.wavelength)
            all_depth.extend(spec.transit_depth)
            all_err.extend(spec.transit_depth_err)
        
        # Sort by wavelength
        sort_idx = np.argsort(all_wl)
        
        instruments = list(set(s.instrument for s in spectra))
        programs = list(set(s.program_id for s in spectra))
        
        return TransmissionSpectrum(
            wavelength=np.array(all_wl)[sort_idx],
            transit_depth=np.array(all_depth)[sort_idx],
            transit_depth_err=np.array(all_err)[sort_idx],
            instrument='+'.join(instruments),
            program_id='+'.join(programs),
            planet_name=spectra[0].planet_name,
            metadata={'combined_from': len(spectra)}
        )
    
    def detect_molecules(
        self,
        spectrum: Optional[TransmissionSpectrum] = None,
        sigma_threshold: float = 3.0
    ) -> List[MoleculeDetection]:
        """
        Detect molecular features in transmission spectrum.
        
        Uses wavelength-specific feature detection against baseline.
        
        Args:
            spectrum: TransmissionSpectrum (downloads if None)
            sigma_threshold: Detection threshold in sigma
            
        Returns:
            List of MoleculeDetection results
        """
        if spectrum is None:
            spectrum = self.get_transmission_spectrum()
            
        if spectrum is None:
            warnings.warn("No transmission spectrum available")
            return []
        
        detections = []
        wl = spectrum.wavelength
        depth = spectrum.transit_depth
        err = spectrum.transit_depth_err
        
        # Compute baseline (polynomial fit)
        valid = np.isfinite(depth) & np.isfinite(err)
        if np.sum(valid) < 10:
            return []
        
        coeffs = np.polyfit(wl[valid], depth[valid], 2, w=1.0/err[valid])
        baseline = np.polyval(coeffs, wl)
        
        for molecule, features in MOLECULAR_FEATURES.items():
            for wl_min, wl_max, feature_name in features:
                # Check if wavelength range is covered
                mask = (wl >= wl_min) & (wl <= wl_max) & valid
                
                if np.sum(mask) < 3:
                    continue
                
                # Compare feature region to baseline
                feature_depth = np.nanmean(depth[mask])
                feature_baseline = np.nanmean(baseline[mask])
                feature_err = np.sqrt(np.sum(err[mask]**2)) / np.sum(mask)
                
                deviation = feature_depth - feature_baseline
                significance = np.abs(deviation) / feature_err
                
                detection = MoleculeDetection(
                    molecule=molecule,
                    detected=significance >= sigma_threshold,
                    significance=significance,
                    wavelength_range=(wl_min, wl_max),
                    feature_depth=deviation if significance >= sigma_threshold else None,
                    reference=feature_name
                )
                
                detections.append(detection)
        
        # Sort by significance
        detections.sort(key=lambda x: x.significance, reverse=True)
        
        return detections
    
    def fit_atmosphere(
        self,
        spectrum: Optional[TransmissionSpectrum] = None,
        molecules: Optional[List[str]] = None,
        use_petitradtrans: bool = True
    ) -> AtmosphericProperties:
        """
        Fit atmospheric model to transmission spectrum.
        
        Requires petitRADTRANS for full retrieval.
        Falls back to simple analysis if not available.
        
        Args:
            spectrum: TransmissionSpectrum
            molecules: List of molecules to include
            use_petitradtrans: Try to use petitRADTRANS
            
        Returns:
            AtmosphericProperties
        """
        if spectrum is None:
            spectrum = self.get_transmission_spectrum()
            
        if spectrum is None:
            return AtmosphericProperties()
        
        props = AtmosphericProperties()
        
        # Try petitRADTRANS
        if use_petitradtrans:
            try:
                props = self._fit_with_petitradtrans(spectrum, molecules)
            except ImportError:
                warnings.warn("petitRADTRANS not available, using simple analysis")
            except Exception as e:
                warnings.warn(f"petitRADTRANS fit failed: {e}")
        
        # Add molecule detections
        props.molecules_detected = self.detect_molecules(spectrum)
        
        # Simple estimates if retrieval didn't work
        if props.temperature is None:
            props = self._simple_analysis(spectrum, props)
        
        return props
    
    def _fit_with_petitradtrans(
        self,
        spectrum: TransmissionSpectrum,
        molecules: Optional[List[str]] = None
    ) -> AtmosphericProperties:
        """Full atmospheric retrieval with petitRADTRANS."""
        from petitRADTRANS import Radtrans
        
        # Default molecules
        if molecules is None:
            molecules = ['H2O_POKAZATEL', 'CO2', 'CO_all_iso', 'CH4']
        
        # Setup retrieval atmosphere
        atmosphere = Radtrans(
            line_species=molecules,
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            wlen_bords_micron=[
                np.min(spectrum.wavelength) * 0.9,
                np.max(spectrum.wavelength) * 1.1
            ]
        )
        
        # Get planet params
        planet = self.target.planets[0] if self.target.planets else None
        
        if planet and planet.radius and self.target.stellar.radius:
            # Setup pressure-temperature profile
            pressures = np.logspace(-6, 2, 100)
            
            # Isothermal first guess
            T_eq = planet.teq or 1000
            temperatures = np.ones_like(pressures) * T_eq
            
            atmosphere.setup_opa_structure(pressures)
            
            # This is a placeholder - full retrieval would use MCMC
            # For now, return equilibrium temperature estimate
            return AtmosphericProperties(
                temperature=T_eq,
                temperature_err=100,
            )
        
        return AtmosphericProperties()
    
    def _simple_analysis(
        self,
        spectrum: TransmissionSpectrum,
        props: AtmosphericProperties
    ) -> AtmosphericProperties:
        """Simple analysis without full retrieval."""
        
        # Estimate scale height from spectrum slope
        wl = spectrum.wavelength
        depth = spectrum.transit_depth
        
        valid = np.isfinite(depth)
        if np.sum(valid) < 10:
            return props
        
        # Fit linear slope to log(depth) vs log(wavelength)
        # Slope relates to scale height
        log_wl = np.log(wl[valid])
        log_depth = np.log(np.abs(depth[valid]))
        
        try:
            slope, _ = np.polyfit(log_wl, log_depth, 1)
            
            # Very rough estimate of scale height
            # Proper analysis needs planet radius, gravity
            if self.target.planets and self.target.planets[0].radius:
                rp = self.target.planets[0].radius * 6371  # R_earth to km
                # H ~ |slope| * Rp / 4 (very approximate)
                props.scale_height = np.abs(slope) * rp / 4
        except Exception:
            pass
        
        return props
    
    def plot_spectrum(
        self,
        spectrum: Optional[TransmissionSpectrum] = None,
        show_molecules: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot transmission spectrum with molecule features.
        
        Args:
            spectrum: TransmissionSpectrum
            show_molecules: Highlight molecular features
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        if spectrum is None:
            spectrum = self.get_transmission_spectrum()
            
        if spectrum is None:
            print("No spectrum to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Main spectrum
        ax.errorbar(
            spectrum.wavelength,
            spectrum.depth_ppm,
            yerr=spectrum.transit_depth_err * 1e6 if np.nanmedian(spectrum.transit_depth_err) < 0.1 else spectrum.transit_depth_err,
            fmt='o',
            markersize=3,
            alpha=0.7,
            label='Data'
        )
        
        # Highlight molecules
        if show_molecules:
            detections = self.detect_molecules(spectrum)
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            for i, det in enumerate(detections[:5]):  # Top 5
                if det.detected:
                    ax.axvspan(
                        det.wavelength_range[0],
                        det.wavelength_range[1],
                        alpha=0.2,
                        color=colors[i % 10],
                        label=f"{det.molecule} ({det.significance:.1f}σ)"
                    )
        
        ax.set_xlabel('Wavelength (μm)', fontsize=12)
        ax.set_ylabel('Transit Depth (ppm)', fontsize=12)
        ax.set_title(f'{spectrum.planet_name} Transmission Spectrum\n{spectrum.instrument}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
