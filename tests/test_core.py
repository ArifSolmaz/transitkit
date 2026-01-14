import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from transitkit.core.models import TransitModelJAX, LimbDarkeningLaw
from transitkit.core.exceptions import TransitKitError


class TestTransitModels:
    """Test transit model implementations."""
    
    def test_mandel_agol_model(self):
        """Test basic Mandel & Agol transit calculation."""
        model = TransitModelJAX()
        
        # Create test data
        time = np.linspace(0, 10, 1000)
        params = {
            't0': 5.0,
            'period': 5.0,
            'rp_over_rs': 0.1,
            'a_over_rs': 15.0,
            'inc': 89.0,
            'u1': 0.5,
            'u2': 0.0
        }
        
        flux = model.compute(time, params)
        
        # Basic sanity checks
        assert flux.shape == (1000,)
        assert np.all(flux >= 0)  # Flux should be positive
        assert np.all(flux <= 1.1)  # Should be around 1
        
        # Should have a dip at t0
        idx_t0 = np.argmin(np.abs(time - 5.0))
        assert flux[idx_t0] < 0.99  # Should be in transit
    
    def test_limb_darkening_laws(self):
        """Test different limb darkening laws."""
        time = np.linspace(0, 10, 100)
        params = {
            't0': 5.0,
            'period': 5.0,
            'rp_over_rs': 0.1,
            'a_over_rs': 15.0,
            'inc': 89.0
        }
        
        # Test each law
        for law in LimbDarkeningLaw:
            model = TransitModelJAX(law=law)
            
            if law == LimbDarkeningLaw.LINEAR:
                params['u1'], params['u2'] = 0.5, 0.0
            elif law == LimbDarkeningLaw.QUADRATIC:
                params['u1'], params['u2'] = 0.5, 0.2
            elif law == LimbDarkeningLaw.NONLINEAR:
                params['u1'], params['u2'], params['u3'], params['u4'] = 0.5, 0.2, 0.1, -0.1
            
            flux = model.compute(time, params)
            assert flux.shape == (100,)
    
    @pytest.mark.slow
    def test_gpu_acceleration(self):
        """Test GPU acceleration (if available)."""
        from transitkit.optimization.accelerated import AcceleratedTransitModel
        
        # Skip if GPU not available
        try:
            model_gpu = AcceleratedTransitModel(use_gpu=True)
        except ImportError:
            pytest.skip("GPU acceleration not available")
        
        model_cpu = AcceleratedTransitModel(use_gpu=False)
        
        # Large dataset for meaningful timing
        time = np.linspace(0, 100, 100000)
        params = {
            't0': 50.0,
            'period': 10.0,
            'rp_over_rs': 0.1,
            'a_over_rs': 15.0,
            'inc': 89.0,
            'u1': 0.5,
            'u2': 0.0
        }
        
        import time as timer
        
        # Time GPU
        start = timer.time()
        flux_gpu = model_gpu.compute(time, params)
        gpu_time = timer.time() - start
        
        # Time CPU
        start = timer.time()
        flux_cpu = model_cpu.compute(time, params)
        cpu_time = timer.time() - start
        
        # Results should be similar
        assert_allclose(flux_gpu, flux_cpu, rtol=1e-5)
        
        print(f"\nGPU time: {gpu_time:.3f}s, CPU time: {cpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")


class TestExceptions:
    """Test exception handling."""
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        model = TransitModelJAX()
        
        with pytest.raises(TransitKitError):
            # Negative radius ratio should raise error
            model.compute(
                np.array([0, 1, 2]),
                {'rp_over_rs': -0.1, 'period': 1, 't0': 0, 'a_over_rs': 10, 'inc': 90}
            )