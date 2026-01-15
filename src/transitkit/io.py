# io.py - Enhanced data handling
"""
Professional data I/O for various formats and missions.
"""

import numpy as np
import pandas as pd
import warnings
from astropy.io import fits
from astropy.table import Table
import lightkurve as lk
import requests
from io import StringIO, BytesIO
import pickle
import json

# Optional h5py import
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False

def load_tess_data_advanced(target, sectors='all', author='SPOC', 
                           cadence='short', quality_bitmask='default',
                           download_dir=None, cache=True):
    """
    Load TESS data with advanced preprocessing.
    
    Parameters
    ----------
    target : str or int
        Target identifier (TIC ID, name, or coordinates)
    sectors : int, list, or 'all'
        TESS sectors to load
    author : str
        Pipeline author ('SPOC', 'QLP', 'TESS-SPOC')
    cadence : str
        Cadence ('short' for 2-min, 'fast' for 20-sec, 'long' for 30-min)
    quality_bitmask : str or int
        Quality flag bitmask
    download_dir : str
        Directory for cached data
    cache : bool
        Whether to cache downloaded data
        
    Returns
    -------
    lc_collection : lightkurve.LightCurveCollection
        Collection of light curves
    """
    # Search for light curves
    search_result = lk.search_lightcurve(
        target, 
        mission='TESS', 
        author=author,
        cadence=cadence
    )
    
    if len(search_result) == 0:
        raise ValueError(f"No TESS data found for target {target}")
    
    # Filter by sectors if specified
    if sectors != 'all':
        if isinstance(sectors, int):
            sectors = [sectors]
        search_result = search_result[np.isin(
            search_result.table['sequence_number'], sectors
        )]
    
    if len(search_result) == 0:
        raise ValueError(f"No data in specified sectors: {sectors}")
    
    # Download light curves
    lc_collection = search_result.download_all(
        quality_bitmask=quality_bitmask,
        download_dir=download_dir
    )
    
    # Standardize preprocessing
    processed_lcs = []
    for lc in lc_collection:
        # Remove NaNs
        lc = lc.remove_nans()
        
        # Use PDCSAP_FLUX if available
        if hasattr(lc, 'PDCSAP_FLUX') and lc.PDCSAP_FLUX is not None:
            lc.flux = lc.PDCSAP_FLUX
            flux_type = 'PDCSAP'
        else:
            flux_type = 'SAP'
        
        # Normalize
        lc = lc.normalize()
        
        # Add metadata
        lc.meta['FLUX_TYPE'] = flux_type
        lc.meta['SECTOR'] = getattr(lc, 'sector', 'Unknown')
        lc.meta['CADENCE'] = cadence
        lc.meta['AUTHOR'] = author
        
        processed_lcs.append(lc)
    
    return processed_lcs


def load_kepler_data(target, quarters='all', campaign=None, **kwargs):
    """
    Load Kepler/K2 data.
    """
    search_result = lk.search_lightcurve(
        target,
        mission='Kepler',
        quarter=quarters,
        campaign=campaign,
        **kwargs
    )
    
    if len(search_result) == 0:
        raise ValueError(f"No Kepler data found for target {target}")
    
    lc_collection = search_result.download_all()
    
    return lc_collection


def load_ground_based_data(filename, format='auto', **kwargs):
    """
    Load ground-based transit data from various formats.
    
    Supported formats:
    - CSV (time, flux, flux_err, airmass, etc.)
    - FITS
    - ESO
    - ASCII tables
    """
    if format == 'auto':
        # Detect format from extension
        if filename.endswith('.csv'):
            format = 'csv'
        elif filename.endswith('.fits') or filename.endswith('.fit'):
            format = 'fits'
        elif filename.endswith('.dat') or filename.endswith('.txt'):
            format = 'ascii'
        elif filename.endswith('.hdf5') or filename.endswith('.h5'):
            format = 'hdf5'
        else:
            raise ValueError(f"Unknown file format: {filename}")
    
    if format == 'csv':
        data = pd.read_csv(filename, **kwargs)
        
        # Standardize column names
        column_map = {
            'time': ['time', 'jd', 'bjd', 'mjd', 't'],
            'flux': ['flux', 'fluxnorm', 'f', 'relflux'],
            'flux_err': ['flux_err', 'error', 'sigma', 'fluxerror'],
            'airmass': ['airmass', 'am', 'sec(z)'],
            'x': ['x', 'xc', 'xcenter'],
            'y': ['y', 'yc', 'ycenter'],
            'fwhm': ['fwhm', 'seeing'],
            'sky': ['sky', 'background', 'skybg']
        }
        
        # Map columns
        for std_name, possible_names in column_map.items():
            for name in possible_names:
                if name in data.columns:
                    data.rename(columns={name: std_name}, inplace=True)
                    break
        
        return data.to_dict('list')
    
    elif format == 'fits':
        with fits.open(filename) as hdul:
            # Try different extensions
            for hdu in hdul:
                if hdu.data is not None:
                    if isinstance(hdu.data, fits.FITS_rec):
                        # Binary table
                        table = Table(hdu.data)
                        return {col: table[col].data for col in table.colnames}
        
        raise ValueError("No table data found in FITS file")
    
    elif format == 'hdf5':
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 files. Install with: pip install h5py")
        with h5py.File(filename, 'r') as f:
            # Convert to dict
            data = {}
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
            
            f.visititems(visitor)
            return data
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_transit_results(results, filename, format='auto'):
    """
    Export transit analysis results to file.
    
    Formats:
    - JSON: For structured data
    - CSV: For tables
    - HDF5: For large datasets
    - Pickle: For Python objects
    """
    if format == 'auto':
        if filename.endswith('.json'):
            format = 'json'
        elif filename.endswith('.csv'):
            format = 'csv'
        elif filename.endswith('.hdf5') or filename.endswith('.h5'):
            format = 'hdf5'
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            format = 'pickle'
        else:
            format = 'json'  # Default
    
    if format == 'json':
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        with open(filename, 'w') as f:
            json.dump(convert(results), f, indent=2)
    
    elif format == 'csv':
        # Flatten results
        flat_results = flatten_dict(results)
        df = pd.DataFrame([flat_results])
        df.to_csv(filename, index=False)
    
    elif format == 'hdf5':
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 files. Install with: pip install h5py")
        with h5py.File(filename, 'w') as f:
            def save_dict_to_hdf5(group, data):
                for key, value in data.items():
                    if isinstance(value, dict):
                        subgroup = group.create_group(key)
                        save_dict_to_hdf5(subgroup, value)
                    elif isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, str)):
                        group.attrs[key] = value
                    elif isinstance(value, list):
                        # Convert list to numpy array
                        group.create_dataset(key, data=np.array(value))
            
            save_dict_to_hdf5(f, results)
    
    elif format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def fetch_exofop_data(tic_id):
    """
    Fetch data from ExoFOP-TESS database.
    """
    base_url = "https://exofop.ipac.caltech.edu/tess"
    
    # Fetch target info
    info_url = f"{base_url}/target.php?id={tic_id}"
    response = requests.get(info_url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch ExoFOP data for TIC {tic_id}")
    
    # Parse HTML for data (simplified)
    # In practice, you'd use proper HTML parsing
    
    return {
        'tic_id': tic_id,
        'exofop_url': info_url,
        'raw_html': response.text[:1000]  # First 1000 chars
    }


def fetch_mast_data(identifier, mission='TESS', data_type='lightcurve'):
    """
    Fetch data from MAST archive.
    """
    from astroquery.mast import Observations
    
    obs_table = Observations.query_object(identifier)
    
    if len(obs_table) == 0:
        raise ValueError(f"No MAST data found for {identifier}")
    
    # Filter by mission
    if mission:
        obs_table = obs_table[obs_table['obs_collection'] == mission]
    
    # Get data products
    data_products = Observations.get_product_list(obs_table)
    
    # Filter by data type
    if data_type == 'lightcurve':
        mask = data_products['productType'] == 'SCIENCE'
    else:
        mask = np.ones(len(data_products), dtype=bool)
    
    data_products = data_products[mask]
    
    # Download
    manifest = Observations.download_products(data_products)
    
    return manifest