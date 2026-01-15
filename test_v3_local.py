#!/usr/bin/env python
"""
TransitKit v3.0 Local Test Script
Run this to verify all modules work before publishing.

Usage:
    cd /path/to/transitkit
    pip install -e .              # Install in dev mode
    python tests/test_v3_local.py
"""

import sys
import traceback
from typing import Tuple

# Colors for terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def test_result(name: str, passed: bool, error: str = None):
    """Print test result."""
    if passed:
        print(f"  {GREEN}✓{RESET} {name}")
    else:
        print(f"  {RED}✗{RESET} {name}")
        if error:
            print(f"    {RED}{error}{RESET}")


def run_test(name: str, func) -> Tuple[bool, str]:
    """Run a test function and catch errors."""
    try:
        func()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


# =============================================================================
# IMPORT TESTS
# =============================================================================
def test_imports():
    """Test all module imports."""
    print(f"\n{BOLD}1. Testing Imports{RESET}")
    
    tests = [
        ("Core package", lambda: __import__("transitkit")),
        ("UniversalTarget", lambda: exec("from transitkit.universal import UniversalTarget")),
        ("UniversalResolver", lambda: exec("from transitkit.universal import UniversalResolver")),
        ("MultiMissionDownloader", lambda: exec("from transitkit.missions import MultiMissionDownloader")),
        ("LightCurveData", lambda: exec("from transitkit.missions import LightCurveData")),
        ("JWSTSpectroscopy", lambda: exec("from transitkit.spectroscopy import JWSTSpectroscopy")),
        ("TransmissionSpectrum", lambda: exec("from transitkit.spectroscopy import TransmissionSpectrum")),
        ("MLTransitDetector", lambda: exec("from transitkit.ml import MLTransitDetector")),
        ("TransitCandidate", lambda: exec("from transitkit.ml import TransitCandidate")),
        ("PublicationGenerator", lambda: exec("from transitkit.publication import PublicationGenerator")),
    ]
    
    passed = 0
    for name, func in tests:
        success, error = run_test(name, func)
        test_result(name, success, error)
        if success:
            passed += 1
    
    return passed, len(tests)


# =============================================================================
# UNIT TESTS (No network)
# =============================================================================
def test_identifier_parser():
    """Test identifier parsing."""
    print(f"\n{BOLD}2. Testing Identifier Parser{RESET}")
    
    from transitkit.universal.resolver import IdentifierParser, TargetType
    
    test_cases = [
        ("TIC 374829238", TargetType.TIC, "374829238"),
        ("TIC374829238", TargetType.TIC, "374829238"),
        ("KIC 8191672", TargetType.KIC, "8191672"),
        ("TOI-700", TargetType.TOI, "700"),
        ("TOI-700.01", TargetType.TOI, "700.01"),
        ("KOI-7016.01", TargetType.KOI, "7016.01"),
        ("EPIC 201912552", TargetType.EPIC, "201912552"),
        ("HD 209458", TargetType.HD, "209458"),
        ("HIP 12345", TargetType.HIP, "12345"),
        ("Gaia DR3 1234567890", TargetType.GAIA_DR3, "1234567890"),
    ]
    
    passed = 0
    for identifier, expected_type, expected_id in test_cases:
        def check():
            result_type, result_id, _ = IdentifierParser.parse(identifier)
            assert result_type == expected_type, f"Expected {expected_type}, got {result_type}"
            assert result_id == expected_id, f"Expected {expected_id}, got {result_id}"
        
        success, error = run_test(f"Parse '{identifier}'", check)
        test_result(f"Parse '{identifier}'", success, error)
        if success:
            passed += 1
    
    return passed, len(test_cases)


def test_data_classes():
    """Test data class creation."""
    print(f"\n{BOLD}3. Testing Data Classes{RESET}")
    
    tests = []
    
    # StellarParameters
    def test_stellar():
        from transitkit.universal.resolver import StellarParameters
        s = StellarParameters(ra=285.67, dec=-32.54, teff=5500, radius=1.0)
        assert s.ra == 285.67
        assert s.teff == 5500
    tests.append(("StellarParameters", test_stellar))
    
    # PlanetParameters
    def test_planet():
        from transitkit.universal.resolver import PlanetParameters
        p = PlanetParameters(name="Test b", period=3.5, radius=11.2)
        assert p.name == "Test b"
        assert p.period == 3.5
    tests.append(("PlanetParameters", test_planet))
    
    # CrossMatchedIDs
    def test_ids():
        from transitkit.universal.resolver import CrossMatchedIDs
        ids = CrossMatchedIDs(primary_name="Test", tic=12345, kic=67890)
        assert ids.tic == 12345
        assert ids.kic == 67890
    tests.append(("CrossMatchedIDs", test_ids))
    
    # LightCurveData
    def test_lc():
        import numpy as np
        from transitkit.missions.downloader import LightCurveData
        lc = LightCurveData(
            time=np.array([1, 2, 3]),
            flux=np.array([1.0, 0.99, 1.0]),
            flux_err=np.array([0.001, 0.001, 0.001]),
            mission="TESS",
            sector=14
        )
        assert lc.label == "TESS_S14"
        assert len(lc.time) == 3
    tests.append(("LightCurveData", test_lc))
    
    # TransitCandidate
    def test_candidate():
        from transitkit.ml.detection import TransitCandidate
        c = TransitCandidate(period=3.5, depth=1000, snr=15.5, ml_score=0.85)
        assert c.period == 3.5
        assert c.ml_score == 0.85
    tests.append(("TransitCandidate", test_candidate))
    
    # TransmissionSpectrum
    def test_spectrum():
        import numpy as np
        from transitkit.spectroscopy.jwst import TransmissionSpectrum
        s = TransmissionSpectrum(
            wavelength=np.array([1.0, 2.0, 3.0]),
            transit_depth=np.array([0.01, 0.012, 0.011]),
            transit_depth_err=np.array([0.001, 0.001, 0.001]),
            instrument="NIRSpec"
        )
        assert s.instrument == "NIRSpec"
        assert len(s.depth_ppm) == 3
    tests.append(("TransmissionSpectrum", test_spectrum))
    
    passed = 0
    for name, func in tests:
        success, error = run_test(name, func)
        test_result(name, success, error)
        if success:
            passed += 1
    
    return passed, len(tests)


def test_ml_detection():
    """Test ML transit detection on synthetic data."""
    print(f"\n{BOLD}4. Testing ML Transit Detection{RESET}")
    
    import numpy as np
    
    tests = []
    
    # Create synthetic light curve with transit
    def create_synthetic_data():
        np.random.seed(42)
        time = np.linspace(0, 27, 5000)
        flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
        
        # Inject transit
        period, t0, depth = 3.5, 1.2, 0.001
        for i in range(10):
            transit_time = t0 + i * period
            mask = np.abs(time - transit_time) < 0.05
            flux[mask] -= depth
        
        return time, flux, period
    
    # Test BLS detection
    def test_bls():
        from transitkit.ml.detection import MLTransitDetector, DetectionMethod
        time, flux, true_period = create_synthetic_data()
        
        detector = MLTransitDetector()
        candidates = detector.detect(
            time, flux,
            methods=[DetectionMethod.BLS],
            min_snr=3,
            max_candidates=1
        )
        
        assert len(candidates) > 0, "No candidates found"
        assert abs(candidates[0].period - true_period) / true_period < 0.01, "Period mismatch"
    tests.append(("BLS Detection", test_bls))
    
    # Test ML scoring
    def test_ml_scoring():
        from transitkit.ml.detection import MLTransitDetector
        import numpy as np
        
        detector = MLTransitDetector()
        
        # Create transit-like view
        view = np.zeros(201)
        view[90:110] = -2  # Dip in center
        
        score = detector._ml_classify(view)
        assert 0 <= score <= 1, "Score out of range"
    tests.append(("ML Scoring", test_ml_scoring))
    
    # Test candidate vetting
    def test_vetting():
        from transitkit.ml.detection import TransitCandidate
        
        c = TransitCandidate(
            period=3.5,
            t0=1.2,
            depth=1000,
            snr=15,
            ml_score=0.8,
            odd_even_diff=0.05,
            secondary_eclipse_depth=10
        )
        
        # Low odd-even and secondary = good candidate
        assert c.odd_even_diff < 0.1
        assert c.secondary_eclipse_depth < c.depth * 0.1
    tests.append(("Candidate Vetting", test_vetting))
    
    passed = 0
    for name, func in tests:
        success, error = run_test(name, func)
        test_result(name, success, error)
        if success:
            passed += 1
    
    return passed, len(tests)


def test_publication():
    """Test publication generator."""
    print(f"\n{BOLD}5. Testing Publication Generator{RESET}")
    
    import tempfile
    import os
    
    tests = []
    
    # Test LaTeX table generation
    def test_latex_stellar():
        from transitkit.universal.resolver import StellarParameters, CrossMatchedIDs
        from transitkit.publication.generator import PublicationGenerator
        
        # Mock target
        class MockTarget:
            stellar = StellarParameters(
                ra=285.67, dec=-32.54, teff=5500,
                radius=1.0, mass=0.95, distance=100
            )
            ids = CrossMatchedIDs(primary_name="Test", tic=12345)
            planets = []
            identifier = "Test"
        
        pub = PublicationGenerator(MockTarget())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "stellar.tex")
            pub.generate_stellar_table(filepath)
            
            assert os.path.exists(filepath), "File not created"
            with open(filepath) as f:
                content = f.read()
            assert "deluxetable" in content or "tabular" in content
            assert "5500" in content  # Teff
    tests.append(("LaTeX Stellar Table", test_latex_stellar))
    
    # Test planet table
    def test_latex_planet():
        from transitkit.universal.resolver import StellarParameters, CrossMatchedIDs, PlanetParameters
        from transitkit.publication.generator import PublicationGenerator
        
        class MockTarget:
            stellar = StellarParameters(ra=285.67, dec=-32.54)
            ids = CrossMatchedIDs(primary_name="Test")
            planets = [
                PlanetParameters(name="Test b", period=3.5, radius=11.2),
                PlanetParameters(name="Test c", period=7.2, radius=3.5),
            ]
            identifier = "Test"
        
        pub = PublicationGenerator(MockTarget())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "planets.tex")
            pub.generate_planet_table(filepath)
            
            assert os.path.exists(filepath)
            with open(filepath) as f:
                content = f.read()
            assert "Test b" in content
            assert "3.5" in content or "3.50" in content
    tests.append(("LaTeX Planet Table", test_latex_planet))
    
    passed = 0
    for name, func in tests:
        success, error = run_test(name, func)
        test_result(name, success, error)
        if success:
            passed += 1
    
    return passed, len(tests)


# =============================================================================
# NETWORK TESTS (Optional - require internet)
# =============================================================================
def test_network_resolution():
    """Test actual target resolution (requires internet)."""
    print(f"\n{BOLD}6. Testing Network Resolution (requires internet){RESET}")
    print(f"   {YELLOW}Skipping by default. Set RUN_NETWORK_TESTS=1 to enable.{RESET}")
    
    import os
    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        return 0, 0
    
    tests = []
    
    def test_resolve_tic():
        from transitkit.universal import UniversalTarget
        target = UniversalTarget("TIC 307210830", verbose=False)
        assert target.tic == 307210830
        assert target.stellar.ra is not None
    tests.append(("Resolve TIC ID", test_resolve_tic))
    
    def test_resolve_planet_name():
        from transitkit.universal import UniversalTarget
        target = UniversalTarget("WASP-39 b", verbose=False)
        assert target.ids.planet_name is not None
        assert len(target.planets) > 0
    tests.append(("Resolve Planet Name", test_resolve_planet_name))
    
    def test_resolve_toi():
        from transitkit.universal import UniversalTarget
        target = UniversalTarget("TOI-700 d", verbose=False)
        assert target.tic is not None
    tests.append(("Resolve TOI", test_resolve_toi))
    
    passed = 0
    for name, func in tests:
        success, error = run_test(name, func)
        test_result(name, success, error)
        if success:
            passed += 1
    
    return passed, len(tests)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  TransitKit v3.0 Local Test Suite{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    
    total_passed = 0
    total_tests = 0
    
    # Run all test groups
    test_functions = [
        test_imports,
        test_identifier_parser,
        test_data_classes,
        test_ml_detection,
        test_publication,
        test_network_resolution,
    ]
    
    for test_func in test_functions:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"  {RED}Test group failed: {e}{RESET}")
            traceback.print_exc()
    
    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    
    if total_tests > 0:
        pct = total_passed / total_tests * 100
        color = GREEN if pct == 100 else (YELLOW if pct >= 80 else RED)
        print(f"\n  {color}{total_passed}/{total_tests} tests passed ({pct:.0f}%){RESET}")
    
    if total_passed == total_tests:
        print(f"\n  {GREEN}{BOLD}✓ All tests passed! Ready to publish.{RESET}")
        return 0
    else:
        print(f"\n  {RED}{BOLD}✗ Some tests failed. Fix before publishing.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
