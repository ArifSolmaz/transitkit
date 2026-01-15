"""Benchmarks for fitting algorithms."""
import time
import numpy as np
from transitkit.core import find_transits_bls_advanced, generate_transit_signal_mandel_agol, add_noise

def benchmark_bls():
    time_arr = np.linspace(0, 100, 10000)
    flux = generate_transit_signal_mandel_agol(time_arr, period=5.0, depth=0.01)
    flux = add_noise(flux, noise_level=0.001)
    
    start = time.time()
    result = find_transits_bls_advanced(time_arr, flux)
    elapsed = time.time() - start
    
    print(f"BLS: {elapsed:.2f}s, Period: {result['period']:.4f}")

if __name__ == "__main__":
    benchmark_bls()
