"""
person5_acoustic
================
Person 5: Acoustic Intelligence + Adaptive Model Router

Part of the Military-Grade Robust Speech Enhancement Architecture.

Person 5 is responsible for:
    - Per-frame acoustic feature extraction
    - Adaptive statistics (running median, MAD, EWMA)
    - Adaptive normalization
    - Acoustic complexity scoring C(t) ∈ [0, 1]
    - Hysteresis + dwell-time model switching logic
    - Router state machine (DTLN ↔ DeepFilterNet)
    - Crossfade controller for smooth model transitions

Person 5 is NOT responsible for:
    - STFT / FFT computation
    - DTLN implementation
    - DeepFilterNet implementation
    - Hardware / ADC / DAC
    - NLMS

Quick start:
    from person5_acoustic import AcousticPipeline, AcousticConfig, AcousticFrame
    import numpy as np

    config = AcousticConfig.development_16khz()
    pipeline = AcousticPipeline(config)

    frame = AcousticFrame(
        frame_index=0,
        sample_rate=16000,
        waveform=np.random.randn(320).astype(np.float32) * 0.1,
    )
    diagnostics = pipeline.process(frame)
    print(diagnostics.complexity_score)
    print(diagnostics.router_decision.active_model)
"""

from .interfaces import (
    AcousticConfig,
    AcousticFrame,
    FeatureVector,
    NormalizedFeatureVector,
    ModelID,
    RouterState,
    RouterDecision,
    PipelineDiagnostics,
    ModelPerformanceHints,
)
from .features import FeatureExtractor
from .adaptive_stats import (
    RunningMedian,
    RunningMAD,
    OutlierDetector,
    EWMA,
    ThresholdedEWMA,
    FeatureAdaptiveStats,
    AdaptiveStatsSnapshot,
)
from .normalization import (
    AdaptiveNormalizer,
    NormalizationCalibration,
    FeatureCalibration,
)
from .complexity import ComplexityEngine, ComplexityResult, ModelPerformanceConfig
from .hysteresis import HysteresisController, HysteresisSignal, AdaptiveThreshold
from .crossfade import CrossfadeController, RampType, blend
from .router import ModelRouter, ModelHealth, RouterConfig
from .pipeline import AcousticPipeline

__version__ = "0.1.0"
__author__ = "Person 5 — Acoustic Intelligence Module"

__all__ = [
    # Config & interfaces
    "AcousticConfig",
    "AcousticFrame",
    "FeatureVector",
    "NormalizedFeatureVector",
    "ModelID",
    "RouterState",
    "RouterDecision",
    "PipelineDiagnostics",
    "ModelPerformanceHints",
    # Features
    "FeatureExtractor",
    # Adaptive stats
    "RunningMedian",
    "RunningMAD",
    "OutlierDetector",
    "EWMA",
    "ThresholdedEWMA",
    "FeatureAdaptiveStats",
    "AdaptiveStatsSnapshot",
    # Normalization
    "AdaptiveNormalizer",
    "NormalizationCalibration",
    "FeatureCalibration",
    # Complexity
    "ComplexityEngine",
    "ComplexityResult",
    "ModelPerformanceConfig",
    # Hysteresis
    "HysteresisController",
    "HysteresisSignal",
    "AdaptiveThreshold",
    # Crossfade
    "CrossfadeController",
    "RampType",
    "blend",
    # Router
    "ModelRouter",
    "ModelHealth",
    "RouterConfig",
    # Pipeline
    "AcousticPipeline",
]
