"""
normalization.py
================
Adaptive feature normalization for Person 5.

Philosophy:
    - Naturally bounded features use their natural bounds ([0,1] directly).
    - Frequency-dependent features use Nyquist as the scale reference.
    - Unbounded / distribution-dependent features use robust calibration:
          z = (x - median) / (MAD + epsilon)
          then map to [0, 1] via a sigmoid-like compression.
    - NO arbitrary hard-coded scales (e.g., "RMS / 0.5").
    - Calibration parameters can be supplied from DEVELOPMENT data only.
      NEVER use test data for calibration.

Output:
    All normalized features are clipped to [0, 1].

Usage:
    normalizer = AdaptiveNormalizer(config)
    norm_fv = normalizer.normalize(feature_vector, stats_snapshots)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .interfaces import (
    AcousticConfig,
    FeatureVector,
    NormalizedFeatureVector,
)
from .adaptive_stats import AdaptiveStatsSnapshot


# ---------------------------------------------------------------------------
# Calibration Data (offline, from development set only)
# ---------------------------------------------------------------------------

@dataclass
class FeatureCalibration:
    """
    Optional calibration parameters for a single feature.

    These should be fitted on a representative DEVELOPMENT / VALIDATION set,
    never on the test set.

    Fields:
        median:  robust center of the feature's distribution on the dev set
        mad:     robust spread of the feature's distribution on the dev set
        clip_sigma: how many (robust) standard deviations to clip at

    If calibration is not supplied, the normalizer falls back to using
    the live adaptive statistics (median/MAD from the running window).
    """
    median: Optional[float] = None
    mad: Optional[float] = None
    clip_sigma: float = 3.0

    @property
    def is_calibrated(self) -> bool:
        return self.median is not None and self.mad is not None


@dataclass
class NormalizationCalibration:
    """
    Full normalization calibration for all features.

    Populate from development data using:
        calibration = NormalizationCalibration.from_dev_features(dev_feature_list)

    Fields that are None fall back to live adaptive statistics.
    """
    rms: FeatureCalibration = field(default_factory=FeatureCalibration)
    zcr: FeatureCalibration = field(default_factory=FeatureCalibration)
    spectral_flux: FeatureCalibration = field(default_factory=FeatureCalibration)
    centroid_var: FeatureCalibration = field(default_factory=FeatureCalibration)
    estimated_snr_db: FeatureCalibration = field(default_factory=FeatureCalibration)

    @classmethod
    def from_dev_features(
        cls,
        features: list[FeatureVector],
    ) -> "NormalizationCalibration":
        """
        Fit calibration parameters from a list of development FeatureVectors.

        Args:
            features: list of FeatureVector from a representative dev set.

        Returns:
            NormalizationCalibration with all fields populated.

        IMPORTANT: Only call this with development data, NEVER test data.
        """
        if not features:
            return cls()

        def _fit(values: list[float]) -> FeatureCalibration:
            arr = np.array(values, dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            return FeatureCalibration(median=med, mad=mad)

        rms_vals = [f.rms for f in features]
        zcr_vals = [f.zcr for f in features]
        flux_vals = [f.spectral_flux for f in features]
        cv_vals = [f.centroid_variation for f in features]
        snr_vals = [f.estimated_snr_db for f in features]

        return cls(
            rms=_fit(rms_vals),
            zcr=_fit(zcr_vals),
            spectral_flux=_fit(flux_vals),
            centroid_var=_fit(cv_vals),
            estimated_snr_db=_fit(snr_vals),
        )


# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    """Hard clip to [0, 1]."""
    return max(0.0, min(1.0, x))


def _robust_normalize(
    x: float,
    median: float,
    mad: float,
    epsilon: float = 1e-8,
    clip_sigma: float = 3.0,
) -> float:
    """
    Robust z-score normalization mapped to [0, 1].

    Algorithm:
        1. Compute z = (x - median) / (MAD + epsilon)
        2. Clip z to [-clip_sigma, +clip_sigma]
        3. Map linearly to [0, 1]:
               (z + clip_sigma) / (2 * clip_sigma)

    This avoids hard-coded feature ranges. The only assumption is that
    the distribution is approximately unimodal and that clip_sigma
    controls how far outliers can push the normalized value.

    Args:
        x:          raw feature value
        median:     robust center
        mad:        Median Absolute Deviation (robust spread)
        epsilon:    floor for MAD (prevents div-by-zero when signal is constant)
        clip_sigma: clipping range in robust standard deviations

    Returns:
        float in [0, 1]
    """
    denom = max(mad, epsilon)
    z = (x - median) / denom
    z_clipped = max(-clip_sigma, min(clip_sigma, z))
    # Map [-clip_sigma, +clip_sigma] → [0, 1]
    normalized = (z_clipped + clip_sigma) / (2.0 * clip_sigma)
    return _clip01(normalized)


def _sigmoid_normalize(
    x: float,
    center: float,
    scale: float,
    epsilon: float = 1e-8,
) -> float:
    """
    Sigmoid-based normalization (alternative to linear robust normalize).

    Maps any real-valued x to (0, 1) using a logistic sigmoid
    centered at `center` with sharpness controlled by `scale`.

        sigma(x) = 1 / (1 + exp(-(x - center) / scale))

    Args:
        x:      raw value
        center: inflection point (typically the median)
        scale:  sharpness parameter (typically proportional to MAD)
        epsilon: floor for scale

    Returns:
        float in (0, 1)
    """
    s = max(scale, epsilon)
    exponent = -(x - center) / s
    # Clamp exponent to avoid overflow
    exponent = max(-500.0, min(500.0, exponent))
    return float(1.0 / (1.0 + math.exp(exponent)))


# ---------------------------------------------------------------------------
# Adaptive Normalizer
# ---------------------------------------------------------------------------

class AdaptiveNormalizer:
    """
    Adaptive feature normalizer for Person 5.

    Applies appropriate normalization to each feature:

        Naturally bounded ([0, 1]):
            - ZCR             → pass-through (already [0, 1])
            - SpectralEntropy → pass-through (H / log(K) already [0, 1])
            - TransientScore  → pass-through (by design [0, 1])
            - CentroidVar     → pass-through (normalized to Nyquist in features.py)

        Robust normalization (unbounded in general):
            - RMS             → robust z-score → [0, 1]
            - SpectralFlux    → robust z-score → [0, 1]
            - SNR_difficulty  → inverted robust z-score (high SNR → low difficulty)

    If offline calibration data is available, uses those parameters.
    Otherwise, falls back to live adaptive statistics (median/MAD from the
    running window supplied by the caller).

    Args:
        config:       AcousticConfig
        calibration:  optional NormalizationCalibration from development data
    """

    def __init__(
        self,
        config: AcousticConfig,
        calibration: Optional[NormalizationCalibration] = None,
    ):
        self._config = config
        self._calibration = calibration or NormalizationCalibration()
        self._epsilon = config.norm_epsilon
        self._clip_sigma = config.norm_clip_sigma

    def set_calibration(self, calibration: NormalizationCalibration) -> None:
        """Replace calibration data (e.g., after offline fitting)."""
        self._calibration = calibration

    def normalize(
        self,
        fv: FeatureVector,
        stats: dict[str, AdaptiveStatsSnapshot],
    ) -> NormalizedFeatureVector:
        """
        Normalize a FeatureVector using adaptive statistics.

        Args:
            fv:    raw FeatureVector from the feature extraction stage
            stats: dict mapping feature name → AdaptiveStatsSnapshot
                   (produced by the complexity engine's adaptive trackers)

        Returns:
            NormalizedFeatureVector with all values in [0, 1]
        """
        outlier_flags: dict[str, bool] = {}
        nfv = NormalizedFeatureVector(frame_index=fv.frame_index)

        # ----------------------------------------------------------------
        # ZCR: naturally bounded [0, 1]
        # ----------------------------------------------------------------
        nfv.zcr_norm = _clip01(fv.zcr)
        outlier_flags["zcr"] = stats.get("zcr", _null_snap()).is_outlier

        # ----------------------------------------------------------------
        # Spectral Entropy: H / log(K) → approximately [0, 1]
        # ----------------------------------------------------------------
        nfv.spectral_entropy_norm = _clip01(fv.spectral_entropy)
        outlier_flags["spectral_entropy"] = (
            stats.get("spectral_entropy", _null_snap()).is_outlier
        )

        # ----------------------------------------------------------------
        # Transient Score: [0, 1] by detector design
        # ----------------------------------------------------------------
        nfv.transient_norm = _clip01(fv.transient_score)
        outlier_flags["transient"] = False

        # ----------------------------------------------------------------
        # Centroid Variation: normalized to Nyquist in features.py → [0, 1]
        # ----------------------------------------------------------------
        nfv.centroid_var_norm = _clip01(fv.centroid_variation)
        outlier_flags["centroid_var"] = (
            stats.get("centroid_var", _null_snap()).is_outlier
        )

        # ----------------------------------------------------------------
        # RMS: robust normalization
        # ----------------------------------------------------------------
        snap_rms = stats.get("rms", _null_snap())
        median_r, mad_r = self._get_calib_or_live(
            self._calibration.rms, snap_rms
        )
        nfv.rms_norm = _robust_normalize(
            fv.rms, median_r, mad_r,
            epsilon=self._epsilon, clip_sigma=self._clip_sigma
        )
        outlier_flags["rms"] = snap_rms.is_outlier

        # ----------------------------------------------------------------
        # Spectral Flux: robust normalization
        # ----------------------------------------------------------------
        snap_flux = stats.get("spectral_flux", _null_snap())
        median_f, mad_f = self._get_calib_or_live(
            self._calibration.spectral_flux, snap_flux
        )
        nfv.spectral_flux_norm = _robust_normalize(
            fv.spectral_flux, median_f, mad_f,
            epsilon=self._epsilon, clip_sigma=self._clip_sigma
        )
        outlier_flags["spectral_flux"] = snap_flux.is_outlier

        # ----------------------------------------------------------------
        # SNR → Difficulty (inverted): high SNR = easy = low difficulty
        #
        # Steps:
        #   1. Normalize SNR robustly.
        #   2. Invert: difficulty = 1 - snr_norm
        # ----------------------------------------------------------------
        snap_snr = stats.get("estimated_snr_db", _null_snap())
        median_s, mad_s = self._get_calib_or_live(
            self._calibration.estimated_snr_db, snap_snr
        )
        snr_norm = _robust_normalize(
            fv.estimated_snr_db, median_s, mad_s,
            epsilon=self._epsilon, clip_sigma=self._clip_sigma
        )
        nfv.snr_difficulty_norm = _clip01(1.0 - snr_norm)  # invert
        outlier_flags["estimated_snr_db"] = snap_snr.is_outlier

        nfv.outlier_flags = outlier_flags
        return nfv

    def _get_calib_or_live(
        self,
        cal: FeatureCalibration,
        snap: "AdaptiveStatsSnapshot",
    ) -> tuple[float, float]:
        """
        Return (median, mad) from calibration if available, else from live stats.
        """
        if cal.is_calibrated:
            return cal.median, cal.mad  # type: ignore[return-value]
        return snap.median, snap.mad


# ---------------------------------------------------------------------------
# Null snapshot helper
# ---------------------------------------------------------------------------

def _null_snap() -> AdaptiveStatsSnapshot:
    """Return a zero-valued snapshot (used when stats are not yet available)."""
    return AdaptiveStatsSnapshot(
        median=0.0, mad=0.0,
        fast_ewma=0.0, slow_ewma=0.0,
        is_outlier=False,
    )
