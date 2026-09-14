"""
model_interface.py
==================
Person 4 — Model Interface Contract and Development Stubs.

This module defines the INTEGRATION CONTRACT between the pipeline and
Person 4's future edge-optimised speech-enhancement models (DTLN variant,
DeepFilterNet variant).

It does NOT implement any neural-network architecture.
It does NOT define LSTM sizes, convolution shapes, attention blocks,
hidden dimensions, number of layers, pruning strategy, quantisation
strategy, or training architecture.

Those decisions belong to Person 4 and will be implemented separately.

===========================================================================
CONTRACT SUMMARY
===========================================================================

Input (per STFT frame):
    D:  np.ndarray, shape=(n_fft,), dtype=complex64 or complex128
        Full complex spectrum produced by Person 3's STFTProcessor.

Output:
    S:  np.ndarray, same shape as D, complex dtype, all values finite.
        Enhanced complex spectrum, passed back to Person 3's ISTFT.

Streaming contract:
    - process(D) is called ONCE PER FRAME.
    - The model MAY maintain internal recurrent / stateful information
      between calls; this is expected for LSTM-based models.
    - reset() is called at the start of a NEW independent audio session,
      NOT between frames of the same session.
    - The pipeline MUST NOT recreate the model object between frames.

Failure contract:
    - If process() raises an exception or returns invalid data the
      pipeline catches the error, marks the model unavailable, and falls
      back through the existing P5 router fallback mechanism.
    - Invalid outputs are defined as:
        * wrong shape
        * non-complex dtype
        * any NaN or Inf value
    - The pipeline must not pass invalid spectra to P3's ISTFT.

Multi-rate support:
    - Models declare which sample rates they support via
      ModelMetadata.supported_sample_rates.
    - The pipeline validates compatibility before processing begins.
    - If a model does not support the requested rate the pipeline falls
      back cleanly; it does NOT silently resample.

Deployment targets (future):
    Raspberry Pi 5 / NVIDIA Jetson Orin Nano or equivalent.
    Real-time edge inference — NOT GPU-only inference.

===========================================================================
CURRENT STATUS
===========================================================================

    DTLNStub        DEVELOPMENT STUB ONLY — not a trained model
    DFNStub         DEVELOPMENT STUB ONLY — not a trained model

Both stubs are kept for integration testing.
They must not be presented as production speech enhancement.

===========================================================================
HOW PERSON 4 REPLACES THE STUBS
===========================================================================

    from integration.model_interface import ModelInterface, ModelMetadata

    class EdgeDTLN(ModelInterface):
        def __init__(self, weights_path: str, sample_rate: int = 16000):
            self._model = load_edge_dtln(weights_path)
            self._sr    = sample_rate
            self.reset()

        @property
        def metadata(self) -> ModelMetadata:
            return ModelMetadata(
                name="EdgeDTLN",
                version="1.0.0",
                supported_sample_rates=[16000, 32000, 48000],
                parameter_count=1_000_000,
                is_trained=True,
                is_stub=False,
                streaming_causal=True,
                lookahead_ms=0.0,
                backend="TFLite",
            )

        def process(self, D: np.ndarray) -> np.ndarray:
            # Person 4 implements this using their trained model
            ...

        def reset(self) -> None:
            # Clear LSTM hidden state etc.
            self._model.reset_state()

Passing the model to the pipeline:
    from integration.pipeline import EndToEndPipeline
    from integration.config   import IntegrationConfig

    pipeline = EndToEndPipeline(
        config     = IntegrationConfig(),
        dtln_model = EdgeDTLN("dtln_edge.tflite"),
        dfn_model  = EdgeDFN("dfn_edge.tflite"),
    )

No changes to P3, P5, router logic, or reconstruction are needed.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


# ===========================================================================
# Model metadata
# ===========================================================================

@dataclass
class ModelMetadata:
    """
    Descriptive metadata for a Person 4 speech-enhancement model.

    All fields are optional — stubs and early models may not know every
    value.  Use None / "not available" for unknown quantities.

    This dataclass intentionally contains NO neural-architecture details
    (no LSTM sizes, no convolution shapes, no attention configuration).
    Those are internal to Person 4's implementation.

    Deployment targets:
        Raspberry Pi 5
        NVIDIA Jetson Orin Nano / equivalent edge platform

    Supported backends (future, informational):
        "CPU"          — pure-Python / NumPy
        "ONNX"         — ONNX Runtime
        "TFLite"       — TensorFlow Lite
        "TorchScript"  — PyTorch / torch.jit
        "TensorRT"     — NVIDIA TensorRT
        "other"        — any other runtime
    """

    # Identity
    name: str = "unnamed"
    version: str = "0.0.0"

    # Multi-rate support
    supported_sample_rates: List[int] = field(
        default_factory=lambda: [16000]
    )
    """
    Sample rates (Hz) this model is designed for.
    Valid values: 16000, 32000, 48000 (or any subset).
    The pipeline validates the session sample rate against this list.
    """

    # Model size (informational — may be None if unknown)
    parameter_count: Optional[int] = None
    """Trainable parameter count, if known."""

    model_size_bytes: Optional[int] = None
    """On-disk model file size, if known."""

    # Latency / streaming properties
    streaming_causal: bool = True
    """
    True if the model operates in causal (streaming, frame-by-frame) mode
    with no future look-ahead.  Required for real-time operation.
    """

    lookahead_ms: float = 0.0
    """
    Look-ahead in milliseconds.
    0 = fully causal.  Non-zero = requires future frames (not real-time).
    """

    expected_latency_ms: Optional[float] = None
    """
    Expected algorithmic latency in ms (window + model latency).
    Not to be confused with hardware RTF — this is platform-independent.
    """

    # Runtime / deployment
    backend: str = "unknown"
    """
    Runtime backend this model uses.
    Examples: "CPU", "ONNX", "TFLite", "TorchScript", "TensorRT", "other".
    """

    # Training status
    is_trained: bool = False
    """True only when this instance holds genuinely trained weights."""

    is_stub: bool = True
    """True for development stubs — clearly indicates not a real model."""

    # Memory usage (informational — may be None)
    peak_memory_bytes: Optional[int] = None
    """Peak runtime memory, if measured."""

    # Free-form notes
    notes: str = ""

    def supports_sample_rate(self, sample_rate: int) -> bool:
        """Return True if this model declares support for ``sample_rate``."""
        return sample_rate in self.supported_sample_rates

    def as_dict(self) -> dict:
        """Serialise metadata to a plain dictionary (for logging/JSON)."""
        return {
            "name": self.name,
            "version": self.version,
            "supported_sample_rates": self.supported_sample_rates,
            "parameter_count": (
                self.parameter_count if self.parameter_count is not None
                else "not available"
            ),
            "model_size_bytes": (
                self.model_size_bytes if self.model_size_bytes is not None
                else "not available"
            ),
            "streaming_causal": self.streaming_causal,
            "lookahead_ms": self.lookahead_ms,
            "expected_latency_ms": (
                self.expected_latency_ms
                if self.expected_latency_ms is not None
                else "not available"
            ),
            "backend": self.backend,
            "is_trained": self.is_trained,
            "is_stub": self.is_stub,
            "peak_memory_bytes": (
                self.peak_memory_bytes
                if self.peak_memory_bytes is not None
                else "not available"
            ),
            "notes": self.notes,
        }


# ===========================================================================
# Per-model performance counters
# ===========================================================================

@dataclass
class ModelPerformanceStats:
    """
    Runtime performance measurements for one model over a session.

    Values are measured from actual wall-clock inference calls.
    They reflect the development-machine execution time and MUST NOT be
    used to estimate Raspberry Pi or Jetson performance.
    """

    frames_processed: int = 0
    total_inference_sec: float = 0.0
    max_frame_latency_sec: float = 0.0
    failures: int = 0

    @property
    def mean_frame_latency_sec(self) -> Optional[float]:
        if self.frames_processed == 0:
            return None
        return self.total_inference_sec / self.frames_processed

    @property
    def mean_frame_latency_ms(self) -> Optional[float]:
        v = self.mean_frame_latency_sec
        return None if v is None else v * 1000.0

    @property
    def max_frame_latency_ms(self) -> float:
        return self.max_frame_latency_sec * 1000.0

    def model_rtf(self, audio_duration_sec: float) -> Optional[float]:
        """Model-only RTF: total_inference_sec / audio_duration_sec."""
        if audio_duration_sec <= 0:
            return None
        return self.total_inference_sec / audio_duration_sec

    def as_dict(self, audio_duration_sec: float = 0.0) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "total_inference_sec": round(self.total_inference_sec, 6),
            "mean_frame_latency_ms": (
                round(self.mean_frame_latency_ms, 4)
                if self.mean_frame_latency_ms is not None
                else "not available"
            ),
            "max_frame_latency_ms": round(self.max_frame_latency_ms, 4),
            "model_rtf": (
                round(self.model_rtf(audio_duration_sec), 4)
                if audio_duration_sec > 0 and
                   self.model_rtf(audio_duration_sec) is not None
                else "not available"
            ),
            "failures": self.failures,
        }

    def reset(self) -> None:
        self.frames_processed = 0
        self.total_inference_sec = 0.0
        self.max_frame_latency_sec = 0.0
        self.failures = 0


# ===========================================================================
# Output validation
# ===========================================================================

class ModelOutputError(Exception):
    """Raised when a model produces invalid output."""


def validate_model_output(
    S: object,
    expected_shape: tuple,
    model_name: str = "model",
) -> np.ndarray:
    """
    Validate a model's output spectrum before passing it to P3 ISTFT.

    Checks:
        - S is a numpy array
        - shape matches expected_shape
        - dtype is complex (complex64 or complex128)
        - all values are finite (no NaN, no Inf)

    Args:
        S:              model output (should be complex np.ndarray)
        expected_shape: shape D had when passed to process()
        model_name:     for error messages

    Returns:
        S cast to complex64 if all checks pass.

    Raises:
        ModelOutputError: if any check fails.
    """
    if not isinstance(S, np.ndarray):
        raise ModelOutputError(
            f"{model_name}: output is not a numpy array "
            f"(got {type(S).__name__!r})"
        )
    if S.shape != expected_shape:
        raise ModelOutputError(
            f"{model_name}: output shape {S.shape} != "
            f"expected {expected_shape}"
        )
    if not np.issubdtype(S.dtype, np.complexfloating):
        raise ModelOutputError(
            f"{model_name}: output dtype {S.dtype} is not complex. "
            "P3 ISTFT requires a complex spectrum."
        )
    if not np.all(np.isfinite(S)):
        n_bad = int(np.sum(~np.isfinite(S)))
        raise ModelOutputError(
            f"{model_name}: output contains {n_bad} non-finite value(s) "
            "(NaN or Inf). Cannot pass to P3 ISTFT."
        )
    return S.astype(np.complex64)


# ===========================================================================
# Abstract model interface
# ===========================================================================

class ModelInterface(ABC):
    """
    Abstract interface for Person 4's speech-enhancement models.

    This is the ONLY contract between the integration pipeline and
    Person 4's model implementations.  The pipeline never inspects model
    internals — it only calls process() and reset().

    Person 3 is the sole owner of STFT/ISTFT.
    Person 5 is the sole owner of complexity estimation and routing.
    Person 4 owns model inference — this interface defines how the
    integration layer calls that inference.

    ----------
    Subclassing
    ----------
    Implement:
        process(D) -> S      mandatory
        metadata   property  mandatory (return a ModelMetadata instance)
        reset()              optional (override if model has internal state)

    Do NOT change the shape or dtype of D.
    Do NOT introduce another STFT, ISTFT, or overlap-add in process().
    Do NOT buffer multiple frames — process() receives exactly one frame.

    ----------
    Streaming
    ----------
    process() is called once per STFT frame (10 ms at 16 kHz with 50%
    overlap).  Models that maintain recurrent state (e.g. LSTM hidden
    state, noise-PSD estimates) should store that state as instance
    variables across calls.

    reset() clears all internal state.  The pipeline calls reset() at the
    start of a new audio session, NOT between frames.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, D: np.ndarray) -> np.ndarray:
        """
        Process one STFT frame.

        Args:
            D: complex array, shape (n_fft,), dtype complex64 or complex128.
               Full complex spectrum from Person 3's STFTProcessor.

        Returns:
            S: enhanced complex array, SAME shape as D, complex dtype,
               all values finite.  Passed to Person 3's ISTFT.

        Raises:
            Any exception: the pipeline will catch it, log it, mark this
            model unavailable, and use the router fallback path.
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return model metadata (identity, capabilities, runtime info)."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers (override as needed)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all internal model state.

        Called by the pipeline at the start of each new audio session.
        NOT called between frames of the same session.
        Override if your model has recurrent state (LSTM hidden state,
        noise PSD estimates, etc.) that must be cleared between sessions.
        """

    # ------------------------------------------------------------------
    # Convenience properties derived from metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.metadata.name

    @property
    def is_stub(self) -> bool:
        """True if this is a development stub, not a trained model."""
        return self.metadata.is_stub

    @property
    def is_trained(self) -> bool:
        """True only if this model holds genuinely trained weights."""
        return self.metadata.is_trained

    def supports_sample_rate(self, sample_rate: int) -> bool:
        """Return True if this model supports the given sample rate."""
        return self.metadata.supports_sample_rate(sample_rate)

    # ------------------------------------------------------------------
    # Instrumented process (called by pipeline — wraps process())
    # ------------------------------------------------------------------

    def process_validated(
        self,
        D: np.ndarray,
        stats: Optional[ModelPerformanceStats] = None,
    ) -> np.ndarray:
        """
        Call process(D), validate the output, and record timing.

        This is called by the pipeline instead of process() directly so
        that validation and performance tracking are centralised.

        If process() raises or returns invalid data:
            - raises ModelOutputError
            - increments stats.failures if stats is not None

        Args:
            D:     complex input spectrum, shape (n_fft,)
            stats: optional performance counter to update

        Returns:
            Validated complex output, shape (n_fft,), dtype complex64
        """
        t0 = time.perf_counter()
        try:
            S = self.process(D)
        except Exception as exc:
            if stats is not None:
                stats.failures += 1
            raise ModelOutputError(
                f"{self.name}: process() raised an exception: {exc}\n"
                + traceback.format_exc()
            ) from exc
        elapsed = time.perf_counter() - t0

        # Validate output
        try:
            S_valid = validate_model_output(S, D.shape, self.name)
        except ModelOutputError:
            if stats is not None:
                stats.failures += 1
            raise

        if stats is not None:
            stats.frames_processed += 1
            stats.total_inference_sec += elapsed
            if elapsed > stats.max_frame_latency_sec:
                stats.max_frame_latency_sec = elapsed

        return S_valid


# ===========================================================================
# DEVELOPMENT STUBS
# ===========================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  DEVELOPMENT STUB ONLY                                                  │
# │                                                                         │
# │  DTLNStub and DFNStub are NOT trained speech-enhancement models.        │
# │  They exist solely so the integration pipeline can be tested end-to-end │
# │  before Person 4 provides real trained models.                          │
# │                                                                         │
# │  They apply a deterministic frequency-dependent gain to the spectrum.   │
# │  They produce NO meaningful noise suppression.                          │
# │                                                                         │
# │  Replace them by passing real ModelInterface implementations to         │
# │  EndToEndPipeline(dtln_model=..., dfn_model=...).                       │
# └─────────────────────────────────────────────────────────────────────────┘

def _stub_gain_mask(n_fft: int, gain: float) -> np.ndarray:
    """
    Conservative frequency-dependent gain mask for stubs.

    Preserves low frequencies (speech fundamentals) and applies
    mild attenuation in the upper band.  This is NOT a noise
    suppression mask — it is a deterministic test fixture.
    """
    mask = np.ones(n_fft, dtype=np.float32)
    low_bin  = max(1, int(0.05 * n_fft))
    high_bin = max(low_bin + 1, int(0.80 * n_fft))
    mask[low_bin:high_bin] = gain
    mask[high_bin:]        = gain * 0.9
    return mask


class DTLNStub(ModelInterface):
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║  DEVELOPMENT STUB ONLY — NOT A TRAINED MODEL                ║
    ║  Slot: DTLN (lightweight low/moderate-complexity path)      ║
    ║  Person 4 will replace this with an edge-optimised DTLN     ║
    ║  variant when trained weights are available.                ║
    ╚══════════════════════════════════════════════════════════════╝

    Applies a deterministic frequency-dependent gain.
    Preserved solely for integration testing.
    """

    _STUB_METADATA = ModelMetadata(
        name="DTLN — DEVELOPMENT STUB",
        version="stub-0.1",
        supported_sample_rates=[16000, 32000, 48000],
        parameter_count=None,
        is_trained=False,
        is_stub=True,
        streaming_causal=True,
        lookahead_ms=0.0,
        backend="numpy-stub",
        notes=(
            "Development stub for integration testing only. "
            "Not a trained model. No noise suppression. "
            "Person 4 will replace with edge-optimised DTLN."
        ),
    )

    def __init__(self, gain: float = 0.95):
        if not (0 < gain <= 1.0):
            raise ValueError(f"gain must be in (0, 1], got {gain}")
        self._gain = float(gain)

    @property
    def metadata(self) -> ModelMetadata:
        return self._STUB_METADATA

    def process(self, D: np.ndarray) -> np.ndarray:
        D = np.asarray(D)
        mask = _stub_gain_mask(len(D), self._gain)
        mag   = np.abs(D)
        phase = np.angle(D)
        return (mag * mask * np.exp(1j * phase)).astype(D.dtype)


class DFNStub(ModelInterface):
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║  DEVELOPMENT STUB ONLY — NOT A TRAINED MODEL                ║
    ║  Slot: DeepFilterNet (robust high-complexity path)          ║
    ║  Person 4 will replace this with an edge-optimised DFN      ║
    ║  variant when trained weights are available.                ║
    ╚══════════════════════════════════════════════════════════════╝

    Applies a deterministic frequency-dependent gain.
    Preserved solely for integration testing.
    """

    _STUB_METADATA = ModelMetadata(
        name="DeepFilterNet — DEVELOPMENT STUB",
        version="stub-0.1",
        supported_sample_rates=[16000, 32000, 48000],
        parameter_count=None,
        is_trained=False,
        is_stub=True,
        streaming_causal=True,
        lookahead_ms=0.0,
        backend="numpy-stub",
        notes=(
            "Development stub for integration testing only. "
            "Not a trained model. No noise suppression. "
            "Person 4 will replace with edge-optimised DeepFilterNet."
        ),
    )

    def __init__(self, gain: float = 0.92):
        if not (0 < gain <= 1.0):
            raise ValueError(f"gain must be in (0, 1], got {gain}")
        self._gain = float(gain)

    @property
    def metadata(self) -> ModelMetadata:
        return self._STUB_METADATA

    def process(self, D: np.ndarray) -> np.ndarray:
        D = np.asarray(D)
        mask = _stub_gain_mask(len(D), self._gain)
        mag   = np.abs(D)
        phase = np.angle(D)
        return (mag * mask * np.exp(1j * phase)).astype(D.dtype)


# ===========================================================================
# Factory
# ===========================================================================

def get_model(model_id_str: str, **kwargs) -> Optional[ModelInterface]:
    """
    Convenience factory to retrieve a stub by name string.

    Returns DTLNStub or DFNStub for integration testing.
    Real Person 4 models are injected via EndToEndPipeline constructor,
    not through this factory.

    Args:
        model_id_str: "DTLN", "DeepFilterNet", or "NONE"

    Returns:
        ModelInterface stub, or None for "NONE".
    """
    models = {
        "DTLN":          DTLNStub,
        "DeepFilterNet": DFNStub,
        "NONE":          None,
    }
    if model_id_str not in models:
        raise ValueError(
            f"Unknown model: {model_id_str!r}. "
            f"Valid options: {list(models.keys())}"
        )
    cls = models[model_id_str]
    if cls is None:
        return None
    return cls(**kwargs)
