"""
pipeline.py
===========
High-level P5 processing pipeline.

Orchestrates the complete per-frame processing flow:

    AcousticFrame (from P3)
        ↓
    FeatureExtractor         → FeatureVector
        ↓
    ComplexityEngine         → (updates adaptive stats, normalizes, computes C(t))
        ↓
    ModelRouter              → RouterDecision
        ↓
    PipelineDiagnostics      (assembled from all stages)

This module is the primary integration point for:
    - Person 3: supplies AcousticFrame per frame
    - Person 4: consumes RouterDecision + crossfade alpha per frame

The pipeline exposes a single `process(frame)` method.

Usage:
    config = AcousticConfig.development_16khz()
    pipeline = AcousticPipeline(config)
    pipeline.set_model_available(ModelID.DEEP_FILTER_NET, True)

    for frame in p3_stream:
        diagnostics = pipeline.process(frame)
        # diagnostics.router_decision → send to P4
        # diagnostics.complexity_score → log/visualize
"""

from __future__ import annotations

from typing import Optional

from .interfaces import (
    AcousticConfig,
    AcousticFrame,
    ModelID,
    PipelineDiagnostics,
)
from .features import FeatureExtractor
from .complexity import ComplexityEngine, ModelPerformanceConfig
from .normalization import NormalizationCalibration
from .router import ModelRouter


class AcousticPipeline:
    """
    Complete Person 5 acoustic intelligence pipeline.

    Manages all sub-components and orchestrates per-frame processing.

    Args:
        config:          AcousticConfig (use .development_16khz() or .production_48khz())
        calibration:     optional NormalizationCalibration from offline dev data
        model_perf:      optional ModelPerformanceConfig from offline experiments
    """

    def __init__(
        self,
        config: AcousticConfig,
        calibration: Optional[NormalizationCalibration] = None,
        model_perf: Optional[ModelPerformanceConfig] = None,
    ):
        self._config = config
        self._feature_extractor = FeatureExtractor(config)
        self._complexity_engine = ComplexityEngine(config, calibration, model_perf)
        self._router = ModelRouter(config)

    def reset(self) -> None:
        """
        Full reset of all pipeline state.

        Call this when starting a new audio session.
        """
        self._feature_extractor.reset()
        self._complexity_engine.reset()
        self._router.reset()

    # ------------------------------------------------------------------
    # Model availability API (forwarded to router)
    # ------------------------------------------------------------------

    def set_model_available(self, model: ModelID, available: bool) -> None:
        """
        Inform the router that a model is available or unavailable.

        Call this when DTLN or DeepFilterNet is loaded/unloaded.
        """
        self._router.set_model_available(model, available)

    def report_model_failure(self, model: ModelID) -> None:
        """Report a model runtime failure to the router."""
        self._router.report_model_failure(model)

    def reset_model_health(self, model: ModelID) -> None:
        """Reset model health after a reload."""
        self._router.reset_model_health(model)

    # ------------------------------------------------------------------
    # Calibration / performance hints (forwarded to complexity engine)
    # ------------------------------------------------------------------

    def set_calibration(self, calibration: NormalizationCalibration) -> None:
        """
        Supply normalization calibration from offline development data.

        MUST be called with development set data only, never test data.
        """
        self._complexity_engine.set_calibration(calibration)

    def set_model_performance(self, perf: ModelPerformanceConfig) -> None:
        """
        Supply model performance hints from offline experiments.

        MUST be filled from offline dev measurements, not from runtime.
        """
        self._complexity_engine.set_model_performance(perf)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(self, frame: AcousticFrame) -> PipelineDiagnostics:
        """
        Process one AcousticFrame through the complete P5 pipeline.

        Args:
            frame: AcousticFrame from Person 3

        Returns:
            PipelineDiagnostics with all per-frame information:
                - features (raw FeatureVector)
                - normalized features
                - complexity score C(t)
                - contributions breakdown
                - router decision (active model, alpha, state)
                - adaptive stats snapshots
                - transient and outlier flags
        """
        # ----------------------------------------------------------------
        # Stage 1: Feature Extraction
        # ----------------------------------------------------------------
        features = self._feature_extractor.process(frame)

        # ----------------------------------------------------------------
        # Stage 2: Adaptive Statistics + Normalization + Complexity
        # ----------------------------------------------------------------
        complexity_result = self._complexity_engine.process(features)

        # ----------------------------------------------------------------
        # Stage 3: Model Routing
        # ----------------------------------------------------------------
        router_decision = self._router.process(
            complexity=complexity_result.score,
            frame_index=frame.frame_index,
        )

        # ----------------------------------------------------------------
        # Assemble diagnostics
        # ----------------------------------------------------------------
        diag = PipelineDiagnostics(
            frame_index=frame.frame_index,
            features=features,
            normalized=complexity_result.normalized_features,
            complexity_score=complexity_result.score,
            complexity_contributions=complexity_result.contributions,
            router_decision=router_decision,
            fast_ewma_snapshot=self._complexity_engine.get_fast_baseline(),
            slow_ewma_snapshot=self._complexity_engine.get_slow_baseline(),
            is_transient=features.is_transient,
            outlier_flags=(
                complexity_result.normalized_features.outlier_flags
                if complexity_result.normalized_features else {}
            ),
            model_availability=self._router.model_availability(),
        )

        return diag

    # ------------------------------------------------------------------
    # Properties / state access
    # ------------------------------------------------------------------

    @property
    def config(self) -> AcousticConfig:
        return self._config

    @property
    def active_model(self) -> ModelID:
        return self._router.active_model

    @property
    def router_state(self):
        return self._router.state

    @property
    def frame_count(self) -> int:
        return self._router.frame_count
