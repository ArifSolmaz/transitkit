# transitkit/fitting/fitters.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional
import emcee
import dynesty
from multiprocessing import Pool
import pickle

class BaseFitter(ABC):
    """Abstract base class for all fitters"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.result = None
        
    @abstractmethod
    def fit(self, initial_guess, **kwargs):
        pass
    
    @abstractmethod
    def get_uncertainties(self):
        pass

class MCMCFitter(BaseFitter):
    """MCMC fitting using emcee"""
    
    def __init__(self, model, data, n_walkers=32):
        super().__init__(model, data)
        self.n_walkers = n_walkers
        self.sampler = None
        
    def ln_likelihood(self, theta):
        """Log likelihood function"""
        # Extract parameters
        params = self._theta_to_params(theta)
        
        # Compute model
        model_flux = self.model.compute(self.data.time, params)
        
        # Calculate chi^2
        residuals = self.data.flux - model_flux
        chi2 = np.sum((residuals / self.data.flux_err) ** 2)
        
        return -0.5 * chi2
    
    def ln_prior(self, theta):
        """Prior probability"""
        params = self._theta_to_params(theta)
        
        # Uniform priors
        if not (0.01 < params['rp_over_rs'] < 0.5):
            return -np.inf
        if not (1.0 < params['a_over_rs'] < 100.0):
            return -np.inf
        if not (80.0 < params['inc'] < 90.0):
            return -np.inf
        if not (0.0 < params['u1'] < 1.0):
            return -np.inf
        if not (0.0 < params['u2'] < 1.0):
            return -np.inf
            
        return 0.0
    
    def ln_prob(self, theta):
        """Total log probability"""
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)
    
    def fit(self, initial_guess, n_steps=5000, n_burn=1000):
        """Run MCMC"""
        n_params = len(initial_guess)
        
        # Initialize walkers
        pos = initial_guess + 1e-4 * np.random.randn(
            self.n_walkers, n_params
        )
        
        # Setup sampler
        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                self.n_walkers, n_params, self.ln_prob, pool=pool
            )
            
            # Run burn-in
            print("Running burn-in...")
            state = self.sampler.run_mcmc(pos, n_burn, progress=True)
            self.sampler.reset()
            
            # Run main chain
            print("Running main chain...")
            self.sampler.run_mcmc(state, n_steps, progress=True)
        
        # Store results
        self.result = {
            'samples': self.sampler.get_chain(),
            'lnprob': self.sampler.get_log_prob(),
            'acceptance': self.sampler.acceptance_fraction,
            'flat_samples': self.sampler.get_chain(discard=n_burn, flat=True)
        }
        
        return self.result
    
    def get_uncertainties(self):
        """Compute parameter uncertainties"""
        flat_samples = self.result['flat_samples']
        
        percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        
        errors = {}
        for i, param_name in enumerate(self.param_names):
            errors[param_name] = {
                'median': percentiles[1, i],
                'lower': percentiles[1, i] - percentiles[0, i],
                'upper': percentiles[2, i] - percentiles[1, i]
            }
        
        return errors

class NestedSamplerFitter(BaseFitter):
    """Nested sampling using dynesty"""
    
    def fit(self, prior_transform, n_live=1000):
        """Run nested sampling"""
        
        sampler = dynesty.NestedSampler(
            self.ln_likelihood,
            prior_transform,
            ndim=len(self.param_names),
            nlive=n_live,
            bound='multi',
            sample='auto'
        )
        
        sampler.run_nested(print_progress=True)
        
        results = sampler.results
        
        self.result = {
            'samples': results.samples,
            'weights': results.weights,
            'logz': results.logz[-1],
            'logzerr': results.logzerr[-1],
            'evidence': np.exp(results.logz[-1])
        }
        
        return self.result

class FittingOrchestrator:
    """Orchestrates multiple fitting strategies"""
    
    def __init__(self):
        self.fitters = {
            'mcmc': MCMCFitter,
            'nested': NestedSamplerFitter,
            'least_squares': LeastSquaresFitter,
            'bayesian_optimization': BayesianOptimizationFitter
        }
        
    def fit_transit(
        self,
        data: LightCurveData,
        method: str = 'mcmc',
        n_planets: int = 1,
        fit_ttv: bool = False,
        **kwargs
    ):
        """Fit transit model with specified method"""
        
        # Initial parameter estimation using BLS
        initial_params = self._bls_initial_guess(data, n_planets)
        
        # Create model
        if n_planets > 1:
            model = MultiTransitModel(n_planets=n_planets)
        else:
            model = TransitModelJAX()
        
        # Choose fitter
        FitterClass = self.fitters[method]
        fitter = FitterClass(model, data)
        
        # Run fit
        result = fitter.fit(initial_params, **kwargs)
        
        # Post-processing
        if fit_ttv:
            result = self._compute_ttv(data, result)
        
        # Quality metrics
        result['metrics'] = self._compute_quality_metrics(data, result)
        
        return result
    
    def _bls_initial_guess(self, data, n_planets):
        """Box Least Squares for initial parameters"""
        from astropy.timeseries import BoxLeastSquares
        
        bls = BoxLeastSquares(data.time, data.flux, data.flux_err)
        
        # Period grid
        duration = np.linspace(0.01, 0.2, 10)  # Fraction of period
        periods = np.linspace(0.5, 100, 10000)
        
        results = []
        for dur in duration:
            bls_result = bls.power(periods, dur, objective='snr')
            results.append(bls_result)
        
        # Find best period for each planet
        initial_params = []
        for i in range(n_planets):
            # Get i-th best period (mask previous ones)
            best_period = self._find_nth_best_period(results, i)
            
            # Fold light curve
            phase = (data.time - data.time[0]) / best_period
            phase = phase - np.floor(phase)
            
            # Estimate depth and duration
            transit_mask = (phase < 0.1) | (phase > 0.9)
            depth = np.median(data.flux[~transit_mask]) - np.median(data.flux[transit_mask])
            
            # Convert to physical parameters
            params = self._depth_to_parameters(depth, best_period)
            initial_params.append(params)
        
        return initial_params