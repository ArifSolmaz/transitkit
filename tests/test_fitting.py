import pytest
import numpy as np
from unittest.mock import Mock, patch

from transitkit.fitting.fitters import MCMCFitter, FittingOrchestrator
from transitkit.data.pipeline import LightCurveData


class TestMCMCFitting:
    """Test MCMC fitting functionality."""
    
    @pytest.fixture
    def mock_light_curve(self):
        """Create mock light curve for testing."""
        time = np.linspace(0, 30, 300)
        flux = np.ones_like(time) + np.random.normal(0, 0.01, len(time))
        
        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=np.ones_like(flux) * 0.01,
            quality=np.zeros_like(flux),
            cadence=0.1,
            mission='TEST',
            target_id='TEST-001'
        )
    
    def test_mcmc_initialization(self, mock_light_curve):
        """Test MCMC fitter initialization."""
        from transitkit.core.models import TransitModelJAX
        
        model = TransitModelJAX()
        fitter = MCMCFitter(model, mock_light_curve, n_walkers=16)
        
        assert fitter.n_walkers == 16
        assert fitter.data is mock_light_curve
        assert fitter.model is model
    
    @pytest.mark.slow
    def test_mcmc_convergence(self, synthetic_transit_light_curve):
        """Test MCMC convergence on synthetic data."""
        from transitkit.core.models import TransitModelJAX
        
        model = TransitModelJAX()
        fitter = MCMCFitter(model, synthetic_transit_light_curve, n_walkers=16)
        
        # Initial guess
        initial_guess = np.array([15.0, 5.0, 0.1, 15.0, 89.0, 0.5, 0.0])
        
        # Run MCMC (reduced steps for testing)
        result = fitter.fit(initial_guess, n_steps=500, n_burn=100)
        
        # Check basic structure
        assert 'samples' in result
        assert 'lnprob' in result
        assert 'acceptance' in result
        
        samples = result['samples']
        assert samples.shape[1] == 7  # 7 parameters
        
        # Check acceptance rate
        acceptance = result['acceptance']
        assert 0.1 < np.mean(acceptance) < 0.9  # Reasonable acceptance
    
    def test_orchestrator(self, mock_light_curve):
        """Test fitting orchestrator."""
        orchestrator = FittingOrchestrator()
        
        # Test available fitters
        assert 'mcmc' in orchestrator.fitters
        assert 'nested' in orchestrator.fitters
        assert 'least_squares' in orchestrator.fitters
        
        # Test with mock fitter
        with patch.object(orchestrator, '_bls_initial_guess') as mock_bls:
            mock_bls.return_value = np.array([0.0, 1.0, 0.05, 10.0, 89.0, 0.3, 0.2])
            
            result = orchestrator.fit_transit(
                mock_light_curve,
                method='least_squares'
            )
            
            assert 'parameters' in result
            assert 'metrics' in result