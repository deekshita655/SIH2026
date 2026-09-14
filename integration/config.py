"""
config.py
=========
Unified configuration for the SIH2026 end-to-end integration pipeline.

Supported sample rates:
    16000 Hz  — current software/development validation target
    32000 Hz  — interface and configuration support (data pending)
    48000 Hz  — production target (hardware data pending)

Window and hop sizes are DERIVED from sample_rate + window_ms + hop_ms.
No size is hard-coded.  For 20 ms window / 10 ms hop:

    16 kHz:  window=320,  hop=160,  n_fft=320
    32 kHz:  window=640,  hop=320,  n_fft=640
    48 kHz:  window=960,  hop=480,  n_fft=960

Dataset status:
    16 kHz: actual development dataset — 1000 noisy/clean pairs
    32 kHz: configuration support only, no dataset yet
    48 kHz: configuration support only, no dataset yet

    Do NOT upsample the 16-kHz dataset to claim 32/48-kHz validation.

Usage:
    config = IntegrationConfig()                          # 16 kHz defaults
    config = IntegrationConfig.development_16khz()        # explicit 16 kHz
    config = IntegrationConfig.for_sample_rate(32000)     # 32 kHz config
    config = IntegrationConfig.production_48khz()         # 48 kHz config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Supported sample rates
# ---------------------------------------------------------------------------

SUPPORTED_SAMPLE_RATES = frozenset({16000, 32000, 48000})
"""
Sample rates the integration layer accepts.

16000:  Phase-1 development dataset.  Full software validation.
32000:  Configuration and interface support.  No dedicated dataset yet.
48000:  Production target.  Requires hardware-recorded dual-mic data.
"""


# ---------------------------------------------------------------------------
# NLMS Configuration (Person 2 parameters)
# ---------------------------------------------------------------------------

@dataclass
class NLMSConfig:
    """
    Configuration for the Python NLMS implementation.

    Defaults match the Person 2 MATLAB reference:
        dsp.LMSFilter('Method','Normalized LMS','Length',64,'StepSize',0.01)
        blockSize = 1024
        Fs_expected = 16000

    All parameters are overridable for experimentation:
        filter_length: try 32, 64 (default), 128
        step_size:     try 0.005, 0.01 (default), 0.05
        block_size:    try 512, 1024 (default), 2048
    """

    # Core NLMS algorithm parameters
    filter_length: int = 64       # NLMS filter length (taps)
    step_size: float = 0.01       # NLMS step size μ ∈ (0, 2]
    block_size: int = 1024        # samples per processing block
    sample_rate: int = 16000      # expected sampling rate (Hz)

    # Numerical stability
    epsilon: float = 1e-8         # denominator regularisation: ||x||² + ε

    # Audio loading options
    convert_to_mono: bool = True  # average stereo channels to mono

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"NLMSConfig.sample_rate must be one of "
                f"{sorted(SUPPORTED_SAMPLE_RATES)}, got {self.sample_rate}. "
                "Note: 32 kHz and 48 kHz are configuration-only; "
                "no validation dataset exists yet for those rates."
            )
        if self.filter_length < 1:
            raise ValueError(f"filter_length must be >= 1, got {self.filter_length}")
        if not (0 < self.step_size <= 2.0):
            raise ValueError(
                f"step_size must be in (0, 2], got {self.step_size}. "
                "For Normalized LMS, values in (0, 1] are typical."
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")


# ---------------------------------------------------------------------------
# DSP Configuration (mirrors Person 3 — integration layer view)
# ---------------------------------------------------------------------------

@dataclass
class DSPIntegrationConfig:
    """
    Person 3 DSP configuration used by the integration layer.

    IMPORTANT: Window and hop sizes are DERIVED from sample_rate +
    window_ms + hop_ms.  Do NOT hard-code 320, 160, 640, 320, 960, 480.
    Person 3's DSPConfig derives them identically using:

        window_length = round(sample_rate * window_ms / 1000)
        hop_length    = round(sample_rate * hop_ms    / 1000)
        n_fft         = window_length  (unless explicitly overridden)

    Derived sizes for 20 ms / 10 ms:
        16 kHz:  window=320,  hop=160,  n_fft=320,  one-sided=161
        32 kHz:  window=640,  hop=320,  n_fft=640,  one-sided=321
        48 kHz:  window=960,  hop=480,  n_fft=960,  one-sided=481

    Dataset status:
        16 kHz: active development dataset
        32 kHz: configuration/interface support only (no dataset)
        48 kHz: configuration/interface support only (no hardware data)
    """

    sample_rate: int   = 16000   # Hz
    window_ms:   float = 20.0    # ms
    hop_ms:      float = 10.0    # ms
    tail_policy: str   = "zero_pad"
    dtype:       str   = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"DSPIntegrationConfig.sample_rate must be one of "
                f"{sorted(SUPPORTED_SAMPLE_RATES)}, got {self.sample_rate}."
            )

    # ------------------------------------------------------------------
    # Derived quantities (no hard-coded constants)
    # ------------------------------------------------------------------

    @property
    def window_length(self) -> int:
        """Derived window length in samples: round(sample_rate × window_ms / 1000)."""
        return round(self.sample_rate * self.window_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        """Derived hop length in samples: round(sample_rate × hop_ms / 1000)."""
        return round(self.sample_rate * self.hop_ms / 1000.0)

    @property
    def n_fft(self) -> int:
        """FFT size: equals window_length (no zero-padding)."""
        return self.window_length

    @property
    def n_onesided(self) -> int:
        """One-sided FFT bins: n_fft // 2 + 1."""
        return self.n_fft // 2 + 1


# ---------------------------------------------------------------------------
# Top-level Integration Configuration
# ---------------------------------------------------------------------------

@dataclass
class IntegrationConfig:
    """
    Unified configuration for the full SIH2026 end-to-end pipeline.

    Sample rates:
        16000 Hz = current software development validation (Phase 1)
        32000 Hz = configuration support (no dedicated dataset yet)
        48000 Hz = production target (requires hardware-recorded data)

    Validation rules:
        nlms.sample_rate must equal dsp.sample_rate.
        Both must be in {16000, 32000, 48000}.

    Model slots:
        dtln_stub_gain  — default gain for DTLNStub (integration testing)
        dfn_stub_gain   — default gain for DFNStub  (integration testing)
        Real models are injected via EndToEndPipeline(dtln_model=...).
    """

    # Sub-configurations
    nlms: NLMSConfig             = field(default_factory=NLMSConfig)
    dsp:  DSPIntegrationConfig   = field(default_factory=DSPIntegrationConfig)

    # Dataset / file paths
    dataset_root:    str           = "."
    primary_file:    Optional[str] = None
    reference_file:  Optional[str] = None

    # Output paths
    output_dir: str = "output/"

    # Model stub gains (for DTLNStub / DFNStub integration testing only)
    dtln_stub_gain: float = 0.95
    dfn_stub_gain:  float = 0.92

    # Post-processing safety
    enable_output_limiter: bool  = True
    """Soft-clip final output to [-1, 1] to prevent hard clipping."""

    # Demo / diagnostic mode
    router_demo_mode: bool = False
    """
    If True, run an additional router validation pass with synthetic
    complexity values.  Clearly labelled — does NOT affect audio output.
    """

    # Plotting
    save_plots:  bool = True
    show_plots:  bool = False

    def __post_init__(self) -> None:
        # Validate sample rate consistency
        if self.nlms.sample_rate != self.dsp.sample_rate:
            raise ValueError(
                f"NLMS sample_rate ({self.nlms.sample_rate}) must match "
                f"DSP sample_rate ({self.dsp.sample_rate})"
            )
        # Validate both are in the supported set (belt + suspenders)
        sr = self.dsp.sample_rate
        if sr not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate {sr} not in supported set "
                f"{sorted(SUPPORTED_SAMPLE_RATES)}"
            )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def sample_rate(self) -> int:
        """The configured sample rate (Hz)."""
        return self.dsp.sample_rate

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def development_16khz(cls) -> "IntegrationConfig":
        """
        Standard 16-kHz development configuration.

        Uses the Phase-1 dataset (1000 noisy/clean pairs at 16 kHz).
        This is the only rate with actual data for software validation.
        """
        return cls(
            nlms=NLMSConfig(sample_rate=16000),
            dsp=DSPIntegrationConfig(sample_rate=16000),
        )

    @classmethod
    def for_sample_rate(cls, sample_rate: int, **kwargs) -> "IntegrationConfig":
        """
        Create a configuration for any supported sample rate.

        Args:
            sample_rate: 16000, 32000, or 48000.
            **kwargs:    additional IntegrationConfig field overrides.

        Note:
            32 kHz and 48 kHz are configuration/interface support only.
            No validation dataset exists at those rates.
            Do NOT upsample the 16-kHz dataset.
        """
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, "
                f"got {sample_rate}"
            )
        # Scale NLMS block_size proportionally so it stays ~64 ms
        base_block = 1024  # at 16 kHz
        block_size = round(base_block * sample_rate / 16000)
        return cls(
            nlms=NLMSConfig(sample_rate=sample_rate, block_size=block_size),
            dsp=DSPIntegrationConfig(sample_rate=sample_rate),
            **kwargs,
        )

    @classmethod
    def production_48khz(cls) -> "IntegrationConfig":
        """
        48-kHz production configuration (hardware target).

        WARNING: Requires hardware-recorded 48-kHz dual-microphone audio.
        Do NOT use by upsampling the 16-kHz development dataset.
        No 48-kHz hardware validation has been performed.
        """
        return cls.for_sample_rate(48000)
