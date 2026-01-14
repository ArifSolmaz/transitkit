# transitkit/core/models/transit.py
import jax.numpy as jnp
from jax import jit, vmap
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class LimbDarkeningLaw(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    NONLINEAR = "nonlinear"
    POWER2 = "power2"

@dataclass
class TransitParameters:
    """Physical parameters for a transit model"""
    t0: float            # Transit epoch [BJD]
    period: float        # Orbital period [days]
    rp_over_rs: float    # Planet-to-star radius ratio
    a_over_rs: float     # Semi-major axis to stellar radius
    inc: float           # Inclination [degrees]
    ecc: float = 0.0     # Eccentricity
    w: float = 90.0      # Argument of periastron [degrees]
    u: Tuple[float, ...] = (0.5, 0.0)  # Limb darkening coefficients
    
class TransitModelJAX:
    """GPU-accelerated transit model using JAX"""
    
    def __init__(self, law: LimbDarkeningLaw = LimbDarkeningLaw.QUADRATIC):
        self.law = law
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Pre-compile JAX functions for speed"""
        self._compute_flux = jit(self._compute_flux_impl)
        self._compute_grad = jit(jax.grad(self._compute_flux_impl, argnums=1))
        self._compute_batch = vmap(self._compute_flux, in_axes=(None, 0, None))
    
    @staticmethod
    def _compute_flux_impl(
        time: jnp.ndarray,
        params: Dict[str, float],
        ld_coeffs: jnp.ndarray
    ) -> jnp.ndarray:
        """JAX implementation of Mandel & Agol 2002"""
        # Extract parameters
        t0 = params['t0']
        period = params['period']
        rp_over_rs = params['rp_over_rs']
        a_over_rs = params['a_over_rs']
        inc_rad = jnp.deg2rad(params['inc'])
        
        # Orbital phase
        phase = (time - t0) / period
        phase = phase - jnp.floor(phase)
        phase = jnp.where(phase > 0.5, phase - 1.0, phase)
        
        # True anomaly (circular approximation for now)
        theta = 2 * jnp.pi * phase
        
        # Projected separation
        z = a_over_rs * jnp.sqrt(
            jnp.sin(theta)**2 + jnp.cos(inc_rad)**2 * jnp.cos(theta)**2
        )
        
        # Limb-darkened transit model
        r = rp_over_rs
        
        # Cases: no transit, partial, full
        flux = jnp.where(
            z > 1 + r,
            1.0,  # No transit
            jnp.where(
                z < 1 - r,
                self._full_transit(r, ld_coeffs),  # Full transit
                self._partial_transit(z, r, ld_coeffs)  # Partial
            )
        )
        
        return flux
    
    def _full_transit(self, r: float, u: jnp.ndarray) -> float:
        """Compute flux during full transit"""
        # Implementation depends on limb darkening law
        if self.law == LimbDarkeningLaw.QUADRATIC:
            u1, u2 = u
            return 1 - r**2 * (1 - u1 - 2*u2) - (4/3)*r**3*u2
        # Add other laws...
    
    def _partial_transit(self, z: float, r: float, u: jnp.ndarray) -> float:
        """Compute flux during partial transit"""
        # Complex geometry - use batman as reference initially
        pass