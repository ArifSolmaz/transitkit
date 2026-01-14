# transitkit/data/pipeline.py
import lightkurve as lk
import numpy as np
from astropy.io import fits
from typing import List, Union, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LightCurveData:
    """Container for light curve data"""
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray
    cadence: float
    mission: str
    target_id: str
    sector: Optional[int] = None
    
class DataPipeline:
    """Unified data access and preprocessing"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize MAST connection
        from astroquery.mast import Observations
        self.mast = Observations
        
    def load_target(
        self,
        identifier: str,
        mission: str = "TESS",
        sector: Optional[int] = None,
        aperture_mask: str = "pipeline",
        detrend_method: str = "cotrending"
    ) -> LightCurveData:
        """
        Load and preprocess light curve for a target
        
        Parameters
        ----------
        identifier : str
            TIC, KIC, or EPIC ID
        mission : str
            'TESS', 'Kepler', 'K2', 'CHEOPS'
        sector : int, optional
            Mission sector/campaign
        aperture_mask : str
            'pipeline', 'threshold', or custom
        detrend_method : str
            'cotrending', 'sff', 'gp', or None
            
        Returns
        -------
        LightCurveData
            Processed light curve
        """
        
        cache_key = f"{mission}_{identifier}_{sector}"
        cache_file = self.cache_dir / f"{cache_key}.h5"
        
        # Check cache
        if cache_file.exists():
            return self._load_from_cache(cache_file)
        
        # Download from archive
        if mission.upper() == "TESS":
            lc_collection = lk.search_lightcurve(
                f"TIC {identifier}", mission="TESS", sector=sector
            ).download_all()
        elif mission.upper() == "KEPLER":
            lc_collection = lk.search_lightcurve(
                f"KIC {identifier}", mission="Kepler"
            ).download_all()
        else:
            raise ValueError(f"Unsupported mission: {mission}")
        
        # Process each light curve
        processed_lcs = []
        for lc in tqdm(lc_collection, desc="Processing sectors"):
            
            # 1. Simple aperture photometry
            if aperture_mask != "pipeline":
                lc = self._custom_aperture(lc, method=aperture_mask)
            
            # 2. Remove outliers
            lc = lc.remove_outliers(sigma_upper=5, sigma_lower=5)
            
            # 3. Detrending
            if detrend_method == "cotrending":
                lc = self._cotrend(lc)
            elif detrend_method == "sff":
                lc = self._sff_detrend(lc)
            elif detrend_method == "gp":
                lc = self._gp_detrend(lc)
            
            # 4. Normalize
            lc = lc.normalize()
            
            processed_lcs.append(lc)
        
        # Combine if multiple sectors
        if len(processed_lcs) > 1:
            combined_lc = lk.LightCurveCollection(processed_lcs).stitch()
        else:
            combined_lc = processed_lcs[0]
        
        # Convert to our format
        data = LightCurveData(
            time=combined_lc.time.value,
            flux=combined_lc.flux.value,
            flux_err=combined_lc.flux_err.value if hasattr(combined_lc.flux_err, 'value') else np.ones_like(combined_lc.flux.value) * 0.001,
            quality=np.zeros_like(combined_lc.flux.value),
            cadence=combined_lc.time[1].value - combined_lc.time[0].value,
            mission=mission,
            target_id=identifier,
            sector=sector
        )
        
        # Cache result
        self._save_to_cache(data, cache_file)
        
        return data
    
    def _cotrend(self, lc):
        """Co-trending basis vectors"""
        # Use lightkurve's CBV correction
        try:
            cbv = lk.Correctors.correct(lc, method='cbv')
            return cbv.corrected_lc
        except:
            return lc
    
    def _gp_detrend(self, lc, kernel='matern32'):
        """Gaussian Process detrending"""
        import celerite2
        from celerite2 import terms
        
        # Create GP model
        if kernel == 'matern32':
            kernel = terms.Matern32Term(sigma=1.0, rho=1.0)
        
        gp = celerite2.GaussianProcess(kernel, mean=np.mean(lc.flux.value))
        gp.compute(lc.time.value, yerr=np.std(lc.flux.value)/10)
        
        # Fit and predict
        flux_smooth = gp.predict(lc.flux.value, lc.time.value, return_var=False)
        
        # Detrend
        lc = lc.copy()
        lc.flux = lc.flux / flux_smooth * np.median(flux_smooth)
        
        return lc