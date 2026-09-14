"""
features.py
===========
Per-frame acoustic feature extraction for Person 5.

This module computes the feature vector f_t from raw frame data.

Computed features:
    - RMS              root mean square energy
    - ZCR              zero crossing rate
    - SpectralFlux     L2 norm of consecutive magnitude spectrum difference
    - SpectralEntropy  normalized spectral entropy (H / log(K))
    - SpectralCentroid in Hz, sample-rate aware
    - CentroidVariation |SC_t - SC_(t-1)| normalized to Nyquist
    - TransientScore   deterministic transient detector
    - EstimatedSNR     lightweight runtime proxy (NOT ground truth)

IMPORTANT:
    - P5 does NOT recompute FFT if magnitude/spectrum is supplied by P3.
    - All frequency-dependent features are sample-rate aware.
    - Zero-energy frames are handled safely.
    - No ML / neural network used.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .interfaces import AcousticConfig, AcousticFrame, FeatureVector


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_log(x: float, epsilon: float = 1e-12) -> float:
    """Numerically safe log."""
    return math.log(max(x, epsilon))


def _rms(waveform: np.ndarray) -> float:
    """
    Root Mean Square of a waveform frame.

        RMS_t = sqrt( (1/N) * sum(x[n]^2) )

    Returns 0.0 for empty arrays.
    """
    n = len(waveform)
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))


def _zcr(waveform: np.ndarray) -> float:
    """
    Zero Crossing Rate of a waveform frame.

        ZCR_t = (1 / (N-1)) * sum( indicator(x[n] * x[n-1] < 0) )

    Naturally bounded in [0, 1].
    Returns 0.0 for frames with fewer than 2 samples.
    """
    n = len(waveform)
    if n < 2:
        return 0.0
    x = waveform.astype(np.float64)
    crossings = np.sum(x[1:] * x[:-1] < 0)
    return float(crossings) / (n - 1)


def _spectral_flux(
    current_mag: np.ndarray,
    previous_mag: Optional[np.ndarray],
) -> float:
    """
    Spectral Flux: L2 norm of the per-bin magnitude difference.

        Flux_t = sqrt( sum_k ( A_t[k] - A_(t-1)[k] )^2 )

    On the first frame (previous_mag is None) returns 0.0.
    """
    if previous_mag is None:
        return 0.0
    diff = current_mag.astype(np.float64) - previous_mag.astype(np.float64)
    return float(np.sqrt(np.sum(diff ** 2)))


def _spectral_entropy(magnitude: np.ndarray) -> float:
    """
    Normalized spectral entropy.

        P_t[k] = |D(k,t)|^2
        p_t[k] = P_t[k] / sum(P_t[k])
        H_t    = -sum( p_t[k] * log(p_t[k]) )
        H_norm = H_t / log(K)

    Returns:
        H_norm in approximately [0, 1].
        0.0 if frame has zero energy (safe handling).
        1.0 if spectrum is perfectly flat (maximum entropy).
    """
    mag = magnitude.astype(np.float64)
    power = mag ** 2
    total_power = np.sum(power)

    if total_power < 1e-12:
        return 0.0  # zero-energy frame

    k = len(mag)
    if k < 2:
        return 0.0

    p = power / total_power
    # Avoid log(0): only compute for p > 0
    nonzero = p > 0
    h = -np.sum(p[nonzero] * np.log(p[nonzero]))
    h_max = math.log(k)  # log(K): maximum entropy for K bins
    return float(h / h_max) if h_max > 0 else 0.0


def _spectral_centroid(
    magnitude: np.ndarray,
    freq_bins: np.ndarray,
) -> float:
    """
    Spectral Centroid in Hz.

        SC_t = sum(f[k] * A_t[k]) / sum(A_t[k])

    Sample-rate aware: uses provided frequency bin array.
    Returns the mean frequency if magnitude sum is zero (safe fallback).
    """
    mag = magnitude.astype(np.float64)
    freq = freq_bins.astype(np.float64)
    total = np.sum(mag)
    if total < 1e-12:
        return float(np.mean(freq))  # fallback: mean frequency
    return float(np.dot(freq, mag) / total)


def _build_freq_bins(fft_size: int, sample_rate: int) -> np.ndarray:
    """
    Build the frequency bin array from FFT parameters.

    Returns shape (fft_size // 2 + 1,) for real-valued FFT (one-sided).
    """
    n_bins = fft_size // 2 + 1
    return np.linspace(0.0, sample_rate / 2.0, n_bins)


# ---------------------------------------------------------------------------
# Transient Detector
# ---------------------------------------------------------------------------

class TransientDetector:
    """
    Deterministic transient / impulsive event detector.

    Uses a ratio of instantaneous vs. local baseline energy/flux to detect
    unusually rapid changes. No ML required.

    Strategy:
        1. Maintain a short EWMA of recent RMS (local energy baseline).
        2. If instantaneous RMS / baseline_RMS > rms_ratio_threshold → transient.
        3. Maintain a short EWMA of recent spectral flux.
        4. If instantaneous flux / baseline_flux > flux_ratio_threshold → transient.
        5. Combine both detectors with OR logic.

    Robust features:
        - EWMA baseline prevents a single loud frame from permanently shifting baseline.
        - Both ratio thresholds are configurable.
        - Returns a continuous score in [0, 1] as well as a boolean flag.
    """

    def __init__(self, config: AcousticConfig):
        self._rms_ratio_thresh = config.transient_rms_ratio
        self._flux_ratio_thresh = config.transient_flux_ratio
        self._alpha = 0.15             # EWMA for local baseline (responsive)
        self._baseline_rms: Optional[float] = None
        self._baseline_flux: Optional[float] = None
        self._epsilon = 1e-8

    def reset(self) -> None:
        """Reset detector state (e.g., on session restart)."""
        self._baseline_rms = None
        self._baseline_flux = None

    def update(self, rms: float, flux: float) -> tuple[float, bool]:
        """
        Update detector with current frame statistics.

        Args:
            rms:  current frame RMS
            flux: current frame spectral flux

        Returns:
            (score, is_transient)
            score:        continuous [0, 1] transient confidence
            is_transient: True if clear transient is detected
        """
        if self._baseline_rms is None:
            # Initialize baseline on first frame
            self._baseline_rms = max(rms, self._epsilon)
            self._baseline_flux = max(flux, self._epsilon)
            return 0.0, False

        # Compute ratios
        rms_ratio = rms / max(self._baseline_rms, self._epsilon)
        flux_ratio = flux / max(self._baseline_flux, self._epsilon)

        # Continuous score: how far above each threshold?
        rms_score = min((rms_ratio / self._rms_ratio_thresh), 1.0)
        flux_score = min((flux_ratio / self._flux_ratio_thresh), 1.0)

        # Combined score: take maximum (OR-like)
        score = float(max(rms_score, flux_score))
        is_transient = (
            rms_ratio > self._rms_ratio_thresh
            or flux_ratio > self._flux_ratio_thresh
        )

        # Update baselines AFTER detection so a transient doesn't shift baseline
        # Use slower alpha to make baseline resistant to impulsive updates
        update_alpha = self._alpha if not is_transient else (self._alpha * 0.1)
        self._baseline_rms = (
            (1 - update_alpha) * self._baseline_rms + update_alpha * rms
        )
        self._baseline_flux = (
            (1 - update_alpha) * self._baseline_flux + update_alpha * flux
        )

        return score, is_transient


# ---------------------------------------------------------------------------
# SNR Estimator (proxy / runtime estimate)
# ---------------------------------------------------------------------------

class RuntimeSNREstimator:
    """
    Lightweight runtime SNR proxy estimator.

    IMPORTANT: This is a PROXY / ESTIMATE only.
    It is NOT ground-truth SNR. It does not have a noise reference signal.

    Approach:
        - Track a slow EWMA of low-RMS frames as noise floor estimate.
        - SNR_proxy = 20 * log10(RMS_signal / RMS_noise_estimate)
        - If external SNR (from P2/NLMS) is provided, use that instead.

    Future integration:
        - P2 can supply a proper NLMS-derived noise estimate.
        - Set frame.external_snr_db to use it.
    """

    def __init__(self, config: AcousticConfig):
        self._slow_alpha = 0.005       # very slow baseline update
        self._fast_alpha = 0.1         # fast signal estimate
        self._noise_floor: Optional[float] = None
        self._signal_level: Optional[float] = None
        self._epsilon = 1e-8

    def reset(self) -> None:
        self._noise_floor = None
        self._signal_level = None

    def update(
        self,
        rms: float,
        external_snr_db: Optional[float] = None,
    ) -> float:
        """
        Update SNR estimate and return current estimate in dB.

        If external_snr_db is provided, return it directly.
        Otherwise compute a proxy estimate.
        """
        if external_snr_db is not None:
            return float(external_snr_db)

        rms = max(rms, self._epsilon)

        if self._noise_floor is None:
            self._noise_floor = rms
            self._signal_level = rms
            return 0.0

        # Update signal level (fast)
        self._signal_level = (
            (1 - self._fast_alpha) * self._signal_level
            + self._fast_alpha * rms
        )

        # Update noise floor only with low-energy frames (proxy for noise periods)
        if rms < self._noise_floor * 2.0:  # heuristic: might be noise-only
            self._noise_floor = (
                (1 - self._slow_alpha) * self._noise_floor
                + self._slow_alpha * rms
            )

        noise = max(self._noise_floor, self._epsilon)
        signal = max(self._signal_level, self._epsilon)
        snr_db = 20.0 * math.log10(signal / noise)
        return float(snr_db)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Per-frame feature extractor for Person 5.

    Computes the 7-element feature vector:
        [RMS, ZCR, SpectralFlux, SpectralEntropy, CentroidVariation,
         TransientScore, EstimatedSNR]

    Usage:
        extractor = FeatureExtractor(config)
        features = extractor.process(frame)
    """

    def __init__(self, config: AcousticConfig):
        self._config = config
        self._prev_magnitude: Optional[np.ndarray] = None
        self._prev_centroid: Optional[float] = None
        self._transient_detector = TransientDetector(config)
        self._snr_estimator = RuntimeSNREstimator(config)
        self._nyquist = config.sample_rate / 2.0

        # Pre-build default freq bins (used when P3 doesn't supply them)
        self._default_freq_bins = _build_freq_bins(
            config.fft_size, config.sample_rate
        )

    def reset(self) -> None:
        """Reset all stateful components (e.g., between sessions)."""
        self._prev_magnitude = None
        self._prev_centroid = None
        self._transient_detector.reset()
        self._snr_estimator.reset()

    def _get_magnitude(self, frame: AcousticFrame) -> Optional[np.ndarray]:
        """
        Return magnitude spectrum from frame.
        P5 does NOT recompute FFT if magnitude is already available.
        Falls back to |complex_stft| if magnitude is absent but complex is present.
        """
        if frame.magnitude is not None:
            return frame.magnitude.astype(np.float64)
        if frame.complex_stft is not None:
            return np.abs(frame.complex_stft).astype(np.float64)
        return None

    def _get_freq_bins(self, frame: AcousticFrame, n_bins: int) -> np.ndarray:
        """Return frequency bin array in Hz."""
        if frame.freq_bins is not None and len(frame.freq_bins) == n_bins:
            return frame.freq_bins.astype(np.float64)
        # Build from config (sample-rate aware, not hard-coded to 16 kHz)
        return _build_freq_bins(self._config.fft_size, frame.sample_rate)[:n_bins]

    def process(self, frame: AcousticFrame) -> FeatureVector:
        """
        Extract features from a single AcousticFrame.

        Args:
            frame: AcousticFrame from P3

        Returns:
            FeatureVector with all 7 features populated.
        """
        fv = FeatureVector(frame_index=frame.frame_index)

        # ----------------------------------------------------------------
        # RMS  (requires waveform)
        # ----------------------------------------------------------------
        if frame.has_waveform:
            fv.rms = _rms(frame.waveform)
        else:
            # Approximate RMS from magnitude if waveform unavailable.
            # This is an approximation via Parseval's theorem.
            mag = self._get_magnitude(frame)
            if mag is not None:
                n = self._config.frame_length or self._config.fft_size
                fv.rms = float(np.sqrt(np.sum(mag ** 2) / max(n, 1)))
            else:
                fv.rms = 0.0

        # ----------------------------------------------------------------
        # Zero Energy check
        # ----------------------------------------------------------------
        fv.is_zero_energy = fv.rms < 1e-9

        # ----------------------------------------------------------------
        # ZCR  (requires waveform)
        # ----------------------------------------------------------------
        if frame.has_waveform:
            fv.zcr = _zcr(frame.waveform)
        else:
            fv.zcr = 0.0  # cannot compute without waveform

        # ----------------------------------------------------------------
        # Spectral features  (require magnitude spectrum)
        # ----------------------------------------------------------------
        magnitude = self._get_magnitude(frame)

        if magnitude is not None:
            n_bins = len(magnitude)
            freq_bins = self._get_freq_bins(frame, n_bins)
            nyquist = frame.sample_rate / 2.0  # sample-rate aware

            # Spectral Flux
            fv.spectral_flux = _spectral_flux(magnitude, self._prev_magnitude)

            # Spectral Entropy (normalized)
            fv.spectral_entropy = _spectral_entropy(magnitude)

            # Spectral Centroid (Hz)
            sc = _spectral_centroid(magnitude, freq_bins)
            fv.spectral_centroid = sc

            # Centroid Variation (normalized to Nyquist)
            if self._prev_centroid is not None:
                raw_var = abs(sc - self._prev_centroid)
                fv.centroid_variation = raw_var / max(nyquist, 1.0)
            else:
                fv.centroid_variation = 0.0

            # Store for next frame
            self._prev_magnitude = magnitude.copy()
            self._prev_centroid = sc

        else:
            # No spectrum available
            fv.spectral_flux = 0.0
            fv.spectral_entropy = 0.0
            fv.spectral_centroid = 0.0
            fv.centroid_variation = 0.0

        # ----------------------------------------------------------------
        # Transient Detector
        # ----------------------------------------------------------------
        score, is_transient = self._transient_detector.update(
            fv.rms, fv.spectral_flux
        )
        fv.transient_score = score
        fv.is_transient = is_transient

        # ----------------------------------------------------------------
        # Estimated Runtime SNR (proxy, NOT ground truth)
        # ----------------------------------------------------------------
        fv.estimated_snr_db = self._snr_estimator.update(
            fv.rms, frame.external_snr_db
        )

        return fv
