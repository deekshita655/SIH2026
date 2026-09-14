"""
complexity.py
=============
Acoustic Complexity Engine for Person 5.

Computes the scalar acoustic complexity score C(t) ∈ [0, 1] from the
normalized feature vector.

    C(t) = w_R * R  +  w_Z * Z  +  w_F * F  +  w_H * H
         +  w_V * V  +  w_T * T  +  w_Q * Q

Where:
    R  = normalized RMS contribution
    Z  = normalized ZCR
    F  = normalized spectral flux
    H  = normalized spectral entropy
    V  = normalized centroid variation
    T  = transient score
    Q  = SNR difficulty (inverted: high SNR → low difficulty)

IMPORTANT:
    - The default weights here are placeholders for testing ONLY.
    - Final weights MUST be calibrated using development/validation data.
    - Do NOT treat these as scientifically validated values.

The complexity engine also:
    - Maintains per-feature adaptive statistics (running MAD + two-timescale EWMA)
    - Detects per-feature outliers
    - Provides contribution breakdown for diagnostics
    - Supports model-performance-aware configuration (offline hints)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .interfaces import (
    AcousticConfig,
    NormalizedFeatureVector,
    FeatureVector,
    ModelPerformanceHints,
)
from .adaptive_stats import FeatureAdaptiveStats, AdaptiveStatsSnapshot
from .normalization import AdaptiveNormalizer, NormalizationCalibration


# ---------------------------------------------------------------------------
# Complexity Result
# ---------------------------------------------------------------------------

@dataclass
class ComplexityResult:
    """
    Output of the complexity engine for a single frame.
    """
    frame_index: int
    score: float                                 # C(t) ∈ [0, 1]
    contributions: dict[str, float] = field(default_factory=dict)
    stats_snapshots: dict[str, AdaptiveStatsSnapshot] = field(default_factory=dict)
    normalized_features: Optional[NormalizedFeatureVector] = None
    is_transient: bool = False


# ---------------------------------------------------------------------------
# Model-Performance-Aware Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelPerformanceConfig:
    """
    Offline development measurements used to inform routing decisions.

    IMPORTANT: These values are NOT collected at runtime.
    They must be filled from offline experiments on the development set:
        - Run DTLN and DeepFilterNet on dev set
        - Measure quality (PESQ, STOI, etc.) and RTF for each
        - Populate this struct and pass it to the ComplexityEngine

    Fields:
        quality_delta:      Q_DFN - Q_DTLN  (positive = DFN is better)
        cost_lambda:        tradeoff weight (higher = prefer cheaper model)
        dtln_rtf:           DTLN real-time factor
        dfn_rtf:            DeepFilterNet real-time factor
        complexity_at_parity: complexity score above which DFN quality exceeds DTLN
    """
    quality_delta: float = 0.0           # Q_DFN - Q_DTLN
    cost_lambda: float = 0.5             # cost/quality tradeoff
    dtln_rtf: float = 0.0               # DTLN RTF from dev measurements
    dfn_rtf: float = 0.0                # DFN RTF from dev measurements
    complexity_at_parity: float = 0.5   # complexity where DFN and DTLN are equal

    @classmethod
    def from_hints(cls, hints: ModelPerformanceHints) -> "ModelPerformanceConfig":
        """Construct from ModelPerformanceHints."""
        quality_delta = hints.dfn_avg_pesq - hints.dtln_avg_pesq
        return cls(
            quality_delta=quality_delta,
            cost_lambda=hints.quality_lambda,
            dtln_rtf=hints.dtln_rtf,
            dfn_rtf=hints.dfn_rtf,
        )

    def utility(self, model: str) -> float:
        """
        Compute routing utility for a given model.

            U = quality - lambda * RTF

        Higher utility = prefer this model.
        NOTE: This is a design placeholder. Final formula requires
        offline validation.
        """
        if model == "DTLN":
            return -self.cost_lambda * self.dtln_rtf  # quality = 0 baseline
        elif model == "DFN":
            return self.quality_delta - self.cost_lambda * self.dfn_rtf
        return 0.0


# ---------------------------------------------------------------------------
# Complexity Engine
# ---------------------------------------------------------------------------

class ComplexityEngine:
    """
    Computes per-frame acoustic complexity score C(t) ∈ [0, 1].

    Internal structure:
        - One FeatureAdaptiveStats instance per feature (for running normalization)
        - AdaptiveNormalizer (uses the live stats)
        - Weighted sum of normalized features

    Weights configuration:
        config.complexity_weights is a dict with keys matching feature names.
        Weights must sum to 1.0.

    Usage:
        engine = ComplexityEngine(config)
        result = engine.process(raw_feature_vector)
    """

    _FEATURE_KEYS = [
        "rms",
        "zcr",
        "spectral_flux",
        "spectral_entropy",
        "centroid_var",
        "transient",
        "snr_difficulty",
    ]

    def __init__(
        self,
        config: AcousticConfig,
        calibration: Optional[NormalizationCalibration] = None,
        model_perf: Optional[ModelPerformanceConfig] = None,
    ):
        self._config = config
        self._weights = dict(config.complexity_weights)  # copy
        self._model_perf = model_perf

        # Per-feature adaptive statistics
        self._stats: dict[str, FeatureAdaptiveStats] = {
            key: FeatureAdaptiveStats(
                fast_alpha=config.fast_ewma_alpha,
                slow_alpha=config.slow_ewma_alpha,
                window=config.median_window,
                k=config.mad_k,
                epsilon=config.mad_epsilon,
            )
            for key in self._FEATURE_KEYS
        }

        self._normalizer = AdaptiveNormalizer(config, calibration)
        self._frame_count: int = 0

    def reset(self) -> None:
        """Reset all adaptive state."""
        for s in self._stats.values():
            s.reset()
        self._frame_count = 0

    def set_calibration(self, calibration: NormalizationCalibration) -> None:
        """Supply offline calibration data (development set only)."""
        self._normalizer.set_calibration(calibration)

    def set_model_performance(self, perf: ModelPerformanceConfig) -> None:
        """Supply offline model-performance measurements."""
        self._model_perf = perf

    def process(self, fv: FeatureVector) -> ComplexityResult:
        """
        Compute complexity score for a single frame.

        Args:
            fv: raw FeatureVector from the feature extraction stage

        Returns:
            ComplexityResult with score ∈ [0, 1] and full diagnostics
        """
        # ----------------------------------------------------------------
        # Step 1: Update adaptive statistics per feature
        # ----------------------------------------------------------------
        # Map raw feature values to the stat tracker keys
        raw_vals: dict[str, float] = {
            "rms":              fv.rms,
            "zcr":              fv.zcr,
            "spectral_flux":    fv.spectral_flux,
            "spectral_entropy": fv.spectral_entropy,
            "centroid_var":     fv.centroid_variation,
            "transient":        fv.transient_score,
            "snr_difficulty":   fv.estimated_snr_db,
        }

        snaps: dict[str, AdaptiveStatsSnapshot] = {}
        for key, val in raw_vals.items():
            snap = self._stats[key].update(val)
            snaps[key] = snap

        # ----------------------------------------------------------------
        # Step 2: Adaptive normalization
        # ----------------------------------------------------------------
        # Map complexity-engine keys to normalization keys
        norm_snaps: dict[str, AdaptiveStatsSnapshot] = {
            "rms":              snaps["rms"],
            "zcr":              snaps["zcr"],
            "spectral_flux":    snaps["spectral_flux"],
            "spectral_entropy": snaps["spectral_entropy"],
            "centroid_var":     snaps["centroid_var"],
            "estimated_snr_db": snaps["snr_difficulty"],
        }
        # transient doesn't need robust normalization (already [0,1])

        norm_fv = self._normalizer.normalize(fv, norm_snaps)

        # ----------------------------------------------------------------
        # Step 3: Weighted sum → C(t)
        # ----------------------------------------------------------------
        components: dict[str, float] = {
            "rms":              norm_fv.rms_norm,
            "zcr":              norm_fv.zcr_norm,
            "spectral_flux":    norm_fv.spectral_flux_norm,
            "spectral_entropy": norm_fv.spectral_entropy_norm,
            "centroid_var":     norm_fv.centroid_var_norm,
            "transient":        norm_fv.transient_norm,
            "snr_difficulty":   norm_fv.snr_difficulty_norm,
        }

        score = 0.0
        contributions: dict[str, float] = {}
        for key in self._FEATURE_KEYS:
            w = self._weights.get(key, 0.0)
            c = components.get(key, 0.0)
            contrib = w * c
            contributions[key] = contrib
            score += contrib

        # Clamp to [0, 1] (floating-point safety)
        score = float(max(0.0, min(1.0, score)))

        self._frame_count += 1

        return ComplexityResult(
            frame_index=fv.frame_index,
            score=score,
            contributions=contributions,
            stats_snapshots=snaps,
            normalized_features=norm_fv,
            is_transient=fv.is_transient,
        )

    @property
    def weights(self) -> dict[str, float]:
        """Current complexity weights (read-only view)."""
        return dict(self._weights)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_slow_baseline(self) -> dict[str, float]:
        """Return slow EWMA baseline for each feature (for diagnostics)."""
        result = {}
        for key, stats in self._stats.items():
            val = stats._slow_ewma.value
            result[key] = float(val) if val is not None else 0.0
        return result

    def get_fast_baseline(self) -> dict[str, float]:
        """Return fast EWMA baseline for each feature (for diagnostics)."""
        result = {}
        for key, stats in self._stats.items():
            val = stats._fast_ewma.value
            result[key] = float(val) if val is not None else 0.0
        return result
