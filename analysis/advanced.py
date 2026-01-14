# transitkit/analysis/advanced.py
class TTVAnalyzer:
    """Transit Timing Variations analysis"""
    
    def __init__(self, data, ephemeris):
        self.data = data
        self.ephemeris = ephemeris  # Dictionary of {epoch: t0}
        
    def compute_ttv(self, method='linear'):
        """Compute TTVs"""
        
        observed_times = self._measure_transit_times()
        expected_times = self._predict_transit_times()
        
        ttv = observed_times - expected_times
        
        # Fit sinusoidal model for dynamical interactions
        if method == 'sinusoidal':
            params = self._fit_sinusoidal_model(ttv)
            results = {
                'ttv': ttv,
                'amplitude': params['amplitude'],
                'period': params['period'],
                'phase': params['phase'],
                'significance': self._compute_significance(ttv)
            }
        else:
            results = {'ttv': ttv}
        
        return results
    
    def _measure_transit_times(self):
        """Precise transit time measurement"""
        # Use Mandel & Agol fits for each transit individually
        times = []
        
        for epoch in range(self.ephemeris['n_transits']):
            # Isolate individual transit
            period = self.ephemeris['period']
            t0_pred = self.ephemeris['t0'] + epoch * period
            
            # Extract ±1.5 transit durations around prediction
            duration = period * self.ephemeris['duration_days']
            mask = (self.data.time > t0_pred - 1.5*duration) & \
                   (self.data.time < t0_pred + 1.5*duration)
            
            transit_data = LightCurveData(
                time=self.data.time[mask],
                flux=self.data.flux[mask],
                flux_err=self.data.flux_err[mask]
            )
            
            # Fit single transit
            fitter = FittingOrchestrator().fit_transit(
                transit_data, method='least_squares'
            )
            
            times.append(fitter['parameters']['t0'])
        
        return np.array(times)

class TransmissionSpectroscopyAnalyzer:
    """Analyze multi-wavelength transits for atmospheric features"""
    
    def __init__(self, light_curves_by_wavelength):
        """
        light_curves_by_wavelength: dict of {wavelength: LightCurveData}
        """
        self.lcs = light_curves_by_wavelength
        self.wavelengths = sorted(light_curves_by_wavelength.keys())
        
    def compute_transmission_spectrum(self):
        """Extract radius vs wavelength"""
        
        radii = []
        errors = []
        
        for wl in self.wavelengths:
            lc = self.lcs[wl]
            
            # Fit transit
            result = FittingOrchestrator().fit_transit(lc, method='mcmc')
            
            radii.append(result['parameters']['rp_over_rs'])
            errors.append(result['errors']['rp_over_rs'])
        
        # Convert to physical units
        star_radius = self._get_stellar_radius()  # From catalog
        planet_radii = np.array(radii) * star_radius * 109.0  # Convert to Earth radii
        
        # Fit atmospheric models
        atmospheric_models = self._fit_atmospheric_models(planet_radii, errors)
        
        return {
            'wavelengths': self.wavelengths,
            'radii': planet_radii,
            'errors': errors,
            'best_fit_model': atmospheric_models['best'],
            'model_comparison': atmospheric_models['comparison']
        }
    
    def _fit_atmospheric_models(self, radii, errors):
        """Fit atmospheric transmission models"""
        
        models = {
            'flat': lambda wl, a: a * np.ones_like(wl),
            'rayleigh': lambda wl, a, b: a + b * (wl**-4),
            'water_absorption': lambda wl, a, b, c: a + b * np.exp(-c * (wl - 1.4)**2),
            'cloudy': lambda wl, a, b, c: a + b * (1 - np.exp(-c * wl))
        }
        
        # Fit each model
        fits = {}
        for name, model in models.items():
            # Use curve_fit
            from scipy.optimize import curve_fit
            try:
                popt, pcov = curve_fit(
                    model, self.wavelengths, radii,
                    sigma=errors, absolute_sigma=True
                )
                fits[name] = {
                    'params': popt,
                    'covariance': pcov,
                    'chi2': np.sum(((radii - model(self.wavelengths, *popt)) / errors)**2)
                }
            except:
                fits[name] = None
        
        # Select best model by BIC
        best_model = min(
            fits.items(),
            key=lambda x: x[1]['chi2'] + np.log(len(radii)) * len(x[1]['params'])
            if x[1] else np.inf
        )
        
        return {
            'fits': fits,
            'best': best_model[0],
            'evidence_ratio': self._compute_bayes_factor(fits)
        }