"""Unified configuration for the SIH2026 end-to-end integration pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .reference_quality import ReferenceQualityConfig

SUPPORTED_SAMPLE_RATES = frozenset({16000, 32000, 48000})

@dataclass
class NLMSConfig:
    filter_length: int = 64
    step_size: float = 0.01
    block_size: int = 1024
    sample_rate: int = 16000
    epsilon: float = 1e-8
    convert_to_mono: bool = True
    reference_quality_enabled: bool = True
    reference_quality: ReferenceQualityConfig = field(default_factory=ReferenceQualityConfig)

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"NLMSConfig.sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, got {self.sample_rate}")
        if self.filter_length < 1:
            raise ValueError(f"filter_length must be >= 1, got {self.filter_length}")
        if not (0 < self.step_size <= 2.0):
            raise ValueError(f"step_size must be in (0, 2], got {self.step_size}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

@dataclass
class DSPIntegrationConfig:
    sample_rate: int = 16000
    window_ms: float = 20.0
    hop_ms: float = 10.0
    tail_policy: str = "zero_pad"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, got {self.sample_rate}")

    @property
    def window_length(self) -> int:
        return round(self.sample_rate * self.window_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000.0)

    @property
    def n_fft(self) -> int:
        return self.window_length

    @property
    def n_onesided(self) -> int:
        return self.n_fft // 2 + 1

@dataclass
class IntegrationConfig:
    nlms: NLMSConfig = field(default_factory=NLMSConfig)
    dsp: DSPIntegrationConfig = field(default_factory=DSPIntegrationConfig)
    dataset_root: str = "."
    primary_file: Optional[str] = None
    reference_file: Optional[str] = None
    output_dir: str = "output/"
    dtln_stub_gain: float = 0.95
    dfn_stub_gain: float = 0.92
    enable_output_limiter: bool = True
    router_demo_mode: bool = False
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self) -> None:
        if self.nlms.sample_rate != self.dsp.sample_rate:
            raise ValueError(f"NLMS sample_rate ({self.nlms.sample_rate}) must match DSP sample_rate ({self.dsp.sample_rate})")
        if self.dsp.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, got {self.dsp.sample_rate}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def sample_rate(self) -> int:
        return self.dsp.sample_rate

    @classmethod
    def development_16khz(cls) -> "IntegrationConfig":
        return cls(nlms=NLMSConfig(sample_rate=16000), dsp=DSPIntegrationConfig(sample_rate=16000))

    @classmethod
    def for_sample_rate(cls, sample_rate: int, **kwargs) -> "IntegrationConfig":
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, got {sample_rate}")
        block_size = round(1024 * sample_rate / 16000)
        return cls(nlms=NLMSConfig(sample_rate=sample_rate, block_size=block_size), dsp=DSPIntegrationConfig(sample_rate=sample_rate), **kwargs)

    @classmethod
    def production_48khz(cls) -> "IntegrationConfig":
        return cls.for_sample_rate(48000)
