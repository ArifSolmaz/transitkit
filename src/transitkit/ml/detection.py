"""
TransitKit v3.0 - ML Transit Detection

Machine learning powered transit detection and vetting:
- CNN-based transit signal classifier
- Ensemble detection (BLS + TLS + ML)
- False positive probability estimation
- Automatic vetting score

Example:
    >>> target = UniversalTarget("TIC 374829238")
    >>> detector = MLTransitDetector(target)
    >>> candidates = detector.find_all()
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
from enum import Enum, auto


@dataclass
class TransitCandidate:
    """A detected transit candidate."""

    period: float  # days
    period_err: Optional[float] = None
    t0: float = 0.0  # BJD
    t0_err: Optional[float] = None
    duration: float = 0.0  # hours
    duration_err: Optional[float] = None
    depth: float = 0.0  # ppm
    depth_err: Optional[float] = None
    snr: float = 0.0

    # Vetting scores
    ml_score: float = 0.0  # 0-1, ML classifier confidence
    bls_power: float = 0.0
    tls_sde: float = 0.0

    # False positive indicators
    odd_even_diff: Optional[float] = None
    secondary_eclipse_depth: Optional[float] = None
    centroid_offset: Optional[float] = None  # arcsec

    # Classification
    disposition: str = "PC"  # PC=Planet Candidate, FP=False Positive, CP=Confirmed Planet
    fp_probability: float = 0.5

    # Derived parameters (if stellar params known)
    rp_rs: Optional[float] = None
    rp_earth: Optional[float] = None
    a_rs: Optional[float] = None
    teq: Optional[float] = None
    insol: Optional[float] = None

    def __repr__(self):
        return (
            f"TransitCandidate(P={self.period:.4f}d, "
            f"depth={self.depth:.0f}ppm, SNR={self.snr:.1f}, "
            f"ML={self.ml_score:.2f})"
        )


class DetectionMethod(Enum):
    """Transit detection methods."""

    BLS = auto()  # Box Least Squares
    TLS = auto()  # Transit Least Squares
    ML = auto()  # Machine Learning
    ENSEMBLE = auto()  # Combined


class MLTransitDetector:
    """
    ML-powered transit detection and vetting.

    Combines classical methods (BLS, TLS) with neural network
    classification for robust planet detection.

    Example:
        >>> target = UniversalTarget("TIC 374829238")
        >>> data = MultiMissionDownloader(target).download_all()
        >>> detector = MLTransitDetector()
        >>> candidates = detector.detect(data.stitch())
    """

    def __init__(self, target=None, model_path: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize ML transit detector.

        Args:
            target: Optional UniversalTarget for context
            model_path: Path to trained model weights
            use_gpu: Use GPU for inference
        """
        self.target = target
        self.model_path = model_path
        self.use_gpu = use_gpu

        self._model = None
        self._stellar_params = None

        if target:
            self._stellar_params = target.stellar

    def detect(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        methods: List[DetectionMethod] = None,
        min_period: float = 0.5,
        max_period: float = 30.0,
        min_snr: float = 5.0,
        max_candidates: int = 10,
    ) -> List[TransitCandidate]:
        """
        Detect transit candidates in light curve.

        Args:
            time: Time array (BJD)
            flux: Normalized flux
            flux_err: Flux errors
            methods: Detection methods to use
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            min_snr: Minimum SNR threshold
            max_candidates: Maximum candidates to return

        Returns:
            List of TransitCandidate objects
        """
        if methods is None:
            methods = [DetectionMethod.BLS, DetectionMethod.TLS, DetectionMethod.ML]

        if flux_err is None:
            flux_err = np.ones_like(flux) * np.nanstd(flux)

        # Clean data
        time, flux, flux_err = self._clean_data(time, flux, flux_err)

        candidates = []

        # BLS detection
        if DetectionMethod.BLS in methods:
            bls_candidates = self._run_bls(time, flux, flux_err, min_period, max_period)
            candidates.extend(bls_candidates)

        # TLS detection
        if DetectionMethod.TLS in methods:
            tls_candidates = self._run_tls(time, flux, flux_err, min_period, max_period)
            candidates.extend(tls_candidates)

        # Merge similar candidates
        candidates = self._merge_candidates(candidates)

        # ML scoring
        if DetectionMethod.ML in methods:
            candidates = self._score_with_ml(candidates, time, flux, flux_err)

        # Filter by SNR
        candidates = [c for c in candidates if c.snr >= min_snr]

        # Compute derived parameters
        if self._stellar_params:
            candidates = self._compute_derived_params(candidates)

        # Vet candidates
        candidates = self._vet_candidates(candidates, time, flux, flux_err)

        # Sort by ML score (or SNR if no ML)
        candidates.sort(key=lambda x: x.ml_score if x.ml_score > 0 else x.snr, reverse=True)

        return candidates[:max_candidates]

    def _clean_data(
        self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove NaNs and outliers."""
        # Remove NaNs
        valid = np.isfinite(time) & np.isfinite(flux)
        time = time[valid]
        flux = flux[valid]
        flux_err = flux_err[valid]

        # Remove outliers (5 sigma)
        median = np.median(flux)
        std = np.std(flux)
        valid = np.abs(flux - median) < 5 * std

        return time[valid], flux[valid], flux_err[valid]

    def _run_bls(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        min_period: float,
        max_period: float,
    ) -> List[TransitCandidate]:
        """Run Box Least Squares transit search."""
        try:
            from astropy.timeseries import BoxLeastSquares

            bls = BoxLeastSquares(time, flux, flux_err)

            # Period grid
            periods = np.linspace(min_period, max_period, 10000)

            # Run BLS with duration array
            durations = np.linspace(0.01, 0.1, 10)
            result = bls.power(periods, durations)

            candidates = []

            # Get best period
            best_idx = np.argmax(result.power)
            best_period = result.period[best_idx]
            best_power = result.power[best_idx]
            best_duration = result.duration[best_idx]
            best_t0 = result.transit_time[best_idx]

            # Get transit model for depth estimation
            model = bls.model(time, best_period, best_duration, best_t0)
            depth = 1.0 - np.min(model)

            # Estimate SNR
            in_transit = model < (1.0 - depth * 0.5)
            n_transit_points = np.sum(in_transit)
            noise = np.nanstd(flux)
            snr = depth / noise * np.sqrt(max(n_transit_points, 1))

            candidate = TransitCandidate(
                period=float(best_period),
                t0=float(best_t0),
                duration=float(best_duration) * 24,  # to hours
                depth=float(depth) * 1e6,  # to ppm
                snr=float(snr),
                bls_power=float(best_power),
            )

            candidates.append(candidate)

            return candidates

        except ImportError:
            warnings.warn("astropy.timeseries not available for BLS")
            return []
        except Exception as e:
            warnings.warn(f"BLS failed: {e}")
            return []

    def _run_tls(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        min_period: float,
        max_period: float,
    ) -> List[TransitCandidate]:
        """Run Transit Least Squares search."""
        try:
            from transitleastsquares import transitleastsquares

            # TLS requires specific format
            model = transitleastsquares(time, flux, flux_err)

            # Stellar parameters if available
            kwargs = {
                "period_min": min_period,
                "period_max": max_period,
                "show_progress_bar": False,
            }

            if self._stellar_params:
                if self._stellar_params.radius:
                    kwargs["R_star"] = self._stellar_params.radius
                if self._stellar_params.mass:
                    kwargs["M_star"] = self._stellar_params.mass

            result = model.power(**kwargs)

            candidates = []

            if result.SDE > 5:  # Minimum detection threshold
                candidate = TransitCandidate(
                    period=result.period,
                    period_err=(
                        result.period_uncertainty if hasattr(result, "period_uncertainty") else None
                    ),
                    t0=result.T0,
                    duration=result.duration * 24,  # to hours
                    depth=(1 - result.depth) * 1e6,  # to ppm (TLS depth is 1 - transit depth)
                    snr=result.snr if hasattr(result, "snr") else result.SDE,
                    tls_sde=result.SDE,
                    rp_rs=result.rp_rs if hasattr(result, "rp_rs") else None,
                    odd_even_diff=(
                        result.odd_even_mismatch if hasattr(result, "odd_even_mismatch") else None
                    ),
                )

                candidates.append(candidate)

            return candidates

        except ImportError:
            warnings.warn("transitleastsquares not available")
            return []
        except Exception as e:
            warnings.warn(f"TLS failed: {e}")
            return []

    def _merge_candidates(
        self, candidates: List[TransitCandidate], period_tolerance: float = 0.01
    ) -> List[TransitCandidate]:
        """Merge candidates with similar periods."""
        if len(candidates) <= 1:
            return candidates

        merged = []
        used = set()

        for i, c1 in enumerate(candidates):
            if i in used:
                continue

            # Find similar candidates
            similar = [c1]
            for j, c2 in enumerate(candidates[i + 1 :], i + 1):
                if j in used:
                    continue
                if np.abs(c1.period - c2.period) / c1.period < period_tolerance:
                    similar.append(c2)
                    used.add(j)

            # Merge: take best SNR, combine scores
            best = max(similar, key=lambda x: x.snr)
            best.bls_power = max(c.bls_power for c in similar)
            best.tls_sde = max(c.tls_sde for c in similar)

            merged.append(best)
            used.add(i)

        return merged

    def _score_with_ml(
        self,
        candidates: List[TransitCandidate],
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> List[TransitCandidate]:
        """Score candidates with ML classifier."""

        for candidate in candidates:
            # Phase fold
            phase = ((time - candidate.t0) % candidate.period) / candidate.period
            phase[phase > 0.5] -= 1

            # Create phase-folded view
            view = self._create_transit_view(phase, flux, candidate.period, candidate.duration / 24)

            # Score with ML model
            score = self._ml_classify(view)
            candidate.ml_score = score

        return candidates

    def _create_transit_view(
        self, phase: np.ndarray, flux: np.ndarray, period: float, duration: float, n_bins: int = 201
    ) -> np.ndarray:
        """Create normalized transit view for ML classification."""
        # Bin around transit
        transit_window = 2 * duration / period  # phase units

        bin_edges = np.linspace(-transit_window, transit_window, n_bins + 1)
        binned_flux = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if np.sum(mask) > 0:
                binned_flux[i] = np.median(flux[mask])
            else:
                binned_flux[i] = 1.0

        # Normalize
        binned_flux = (binned_flux - np.median(binned_flux)) / np.std(binned_flux)

        return binned_flux

    def _ml_classify(self, view: np.ndarray) -> float:
        """
        Classify transit view with neural network.

        If no trained model, use heuristic scoring.
        """
        if self._model is not None:
            # Use trained model
            try:
                view_input = view.reshape(1, -1, 1)
                prediction = self._model.predict(view_input, verbose=0)
                return float(prediction[0, 0])
            except Exception:
                pass

        # Heuristic scoring (no ML model)
        score = 0.5

        # Check for transit-like dip
        center = len(view) // 2
        transit_region = view[center - 10 : center + 10]
        out_of_transit = np.concatenate([view[: center - 20], view[center + 20 :]])

        if len(transit_region) > 0 and len(out_of_transit) > 0:
            dip = np.mean(out_of_transit) - np.mean(transit_region)

            # Transit should have a dip
            if dip > 1.0:
                score += 0.2
            if dip > 2.0:
                score += 0.1

            # Should be U-shaped
            if transit_region[0] > transit_region[len(transit_region) // 2]:
                score += 0.1
            if transit_region[-1] > transit_region[len(transit_region) // 2]:
                score += 0.1

        return min(score, 1.0)

    def _compute_derived_params(self, candidates: List[TransitCandidate]) -> List[TransitCandidate]:
        """Compute derived planetary parameters."""
        stellar = self._stellar_params

        for c in candidates:
            # Rp/Rs from depth
            if c.depth > 0:
                c.rp_rs = np.sqrt(c.depth / 1e6)

            # Rp in Earth radii
            if c.rp_rs and stellar.radius:
                c.rp_earth = c.rp_rs * stellar.radius * 109.2  # Rsun to Rearth

            # Semi-major axis from Kepler's law
            if stellar.mass and c.period:
                # a^3 = GM*P^2/(4*pi^2)
                G = 6.674e-11
                M_sun = 1.989e30
                M_star = stellar.mass * M_sun
                P_sec = c.period * 86400

                a_m = (G * M_star * P_sec**2 / (4 * np.pi**2)) ** (1 / 3)
                R_sun = 6.96e8

                if stellar.radius:
                    c.a_rs = a_m / (stellar.radius * R_sun)

            # Equilibrium temperature
            if stellar.teff and c.a_rs:
                # Teq = Teff * sqrt(Rs / (2*a)) * (1-A)^0.25
                # Assume albedo A = 0.3
                c.teq = stellar.teff * np.sqrt(1 / (2 * c.a_rs)) * (0.7) ** 0.25

            # Insolation
            if stellar.luminosity and c.a_rs and stellar.radius:
                a_au = c.a_rs * stellar.radius * 0.00465  # Rs to AU
                c.insol = stellar.luminosity / (a_au**2)

        return candidates

    def _vet_candidates(
        self,
        candidates: List[TransitCandidate],
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> List[TransitCandidate]:
        """Vet candidates for false positive indicators."""

        for c in candidates:
            # Odd-even test
            if c.odd_even_diff is None:
                c.odd_even_diff = self._odd_even_test(time, flux, c)

            # Secondary eclipse check
            c.secondary_eclipse_depth = self._check_secondary(time, flux, c)

            # Estimate FP probability
            c.fp_probability = self._estimate_fp_prob(c)

            # Set disposition
            if c.fp_probability < 0.1 and c.ml_score > 0.7 and c.snr > 10:
                c.disposition = "PC"  # Strong planet candidate
            elif c.fp_probability > 0.5 or c.ml_score < 0.3:
                c.disposition = "FP"  # Likely false positive
            else:
                c.disposition = "PC"  # Uncertain

        return candidates

    def _odd_even_test(
        self, time: np.ndarray, flux: np.ndarray, candidate: TransitCandidate
    ) -> float:
        """Test for odd-even transit depth difference (EB indicator)."""
        try:
            # Phase fold
            phase = ((time - candidate.t0) % candidate.period) / candidate.period
            transit_mask = np.abs(phase - 0.5) < candidate.duration / 24 / candidate.period

            # Separate odd and even transits
            transit_number = np.floor((time - candidate.t0) / candidate.period)
            odd_mask = transit_mask & (transit_number % 2 == 1)
            even_mask = transit_mask & (transit_number % 2 == 0)

            odd_depth = 1 - np.median(flux[odd_mask]) if np.sum(odd_mask) > 0 else 0
            even_depth = 1 - np.median(flux[even_mask]) if np.sum(even_mask) > 0 else 0

            if odd_depth + even_depth > 0:
                return np.abs(odd_depth - even_depth) / ((odd_depth + even_depth) / 2)
            return 0

        except Exception:
            return 0

    def _check_secondary(
        self, time: np.ndarray, flux: np.ndarray, candidate: TransitCandidate
    ) -> float:
        """Check for secondary eclipse (EB indicator)."""
        try:
            # Phase fold
            phase = ((time - candidate.t0) % candidate.period) / candidate.period

            # Check at phase 0.5
            secondary_mask = np.abs(phase - 0.5) < candidate.duration / 24 / candidate.period
            out_mask = (np.abs(phase) > 0.3) & (np.abs(phase) < 0.7) & (~secondary_mask)

            if np.sum(secondary_mask) > 0 and np.sum(out_mask) > 0:
                secondary_depth = np.median(flux[out_mask]) - np.median(flux[secondary_mask])
                return max(0, secondary_depth * 1e6)  # ppm

            return 0

        except Exception:
            return 0

    def _estimate_fp_prob(self, candidate: TransitCandidate) -> float:
        """Estimate false positive probability."""
        fp_prob = 0.5  # Prior

        # ML score
        if candidate.ml_score > 0.8:
            fp_prob -= 0.2
        elif candidate.ml_score < 0.4:
            fp_prob += 0.2

        # SNR
        if candidate.snr > 20:
            fp_prob -= 0.1
        elif candidate.snr < 7:
            fp_prob += 0.1

        # Odd-even
        if candidate.odd_even_diff and candidate.odd_even_diff > 0.3:
            fp_prob += 0.2

        # Secondary eclipse
        if candidate.secondary_eclipse_depth:
            if candidate.secondary_eclipse_depth > candidate.depth * 0.1:
                fp_prob += 0.3  # Likely EB

        # BLS and TLS agreement
        if candidate.bls_power > 0 and candidate.tls_sde > 0:
            fp_prob -= 0.1

        return max(0, min(1, fp_prob))

    def build_cnn_model(self, input_shape: tuple = (201, 1)) -> "Any":
        """
        Build CNN model for transit classification.

        Returns compiled Keras model.
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential(
                [
                    layers.Conv1D(32, 7, activation="relu", input_shape=input_shape),
                    layers.MaxPooling1D(2),
                    layers.Conv1D(64, 5, activation="relu"),
                    layers.MaxPooling1D(2),
                    layers.Conv1D(128, 3, activation="relu"),
                    layers.GlobalAveragePooling1D(),
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])

            return model

        except ImportError:
            warnings.warn("TensorFlow not available")
            return None


# Convenience function
def detect_transits(
    time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None, target=None, **kwargs
) -> List[TransitCandidate]:
    """
    Convenience function for transit detection.

    Args:
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        target: Optional UniversalTarget
        **kwargs: Passed to MLTransitDetector.detect()

    Returns:
        List of TransitCandidate
    """
    detector = MLTransitDetector(target)
    return detector.detect(time, flux, flux_err, **kwargs)
