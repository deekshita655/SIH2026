"""
integration
===========
SIH2026 End-to-End Integration Layer.

Connects Person 2 (NLMS) -> Person 3 (STFT/ISTFT) -> Person 5 (Acoustic
Intelligence + Router) -> Person 4 (model slot) -> Person 3 (ISTFT).

This package contains ONLY integration/adaptor code.
It does NOT replace or duplicate functionality owned by Person 3 or Person 5.

Architecture ownership:
    PERSON 2:  nlms.py            — NLMS adaptive noise cancellation
    PERSON 3:  person3_dsp/       — STFT, ISTFT, framing, windowing, OLA
    PERSON 5:  person5_acoustic/  — Features, complexity, routing, crossfade
    PERSON 4:  model_interface.py — Model slots (stubs / future trained models)

Integration layer:
    config.py          — Unified IntegrationConfig (16/32/48 kHz)
    dataset.py         — Dataset loader / example selector
    pipeline.py        — End-to-end orchestrator with fail-safe model handling
    metrics.py         — Metric collection (SNR, SI-SDR, STOI, PESQ infra)
    plots.py           — Diagnostic plotting
    batch.py           — Optional batch validation mode

Supported sample rates:
    16000 Hz — Phase-1 development dataset (actual data available)
    32000 Hz — Configuration and interface support (no dataset yet)
    48000 Hz — Production target (requires hardware dual-mic recordings)

P4 model slots (current status):
    DTLNStub          — DEVELOPMENT STUB ONLY (no trained model)
    DFNStub           — DEVELOPMENT STUB ONLY (no trained model)
    Person 4 injects real models via EndToEndPipeline(dtln_model=..., dfn_model=...)
"""

from .config import IntegrationConfig, NLMSConfig, SUPPORTED_SAMPLE_RATES
from .nlms import NLMSFilter, NLMSResult
from .dataset import DatasetExample, DatasetLoader
from .model_interface import ModelInterface, ModelMetadata

__version__ = "0.2.0"
__all__ = [
    "IntegrationConfig",
    "NLMSConfig",
    "SUPPORTED_SAMPLE_RATES",
    "NLMSFilter",
    "NLMSResult",
    "DatasetExample",
    "DatasetLoader",
    "ModelInterface",
    "ModelMetadata",
]
