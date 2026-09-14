"""NLMS adaptive noise cancellation with reference-quality gating."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import NLMSConfig
from .reference_quality import ReferenceQualityConfig, ReferenceQualityResult, estimate_reference_quality


@dataclass
class NLMSResult:
    cleaned_speech: np.ndarray
    estimated_noise: np.ndarray
    sample_rate: int
    filter_length: int
    step_size: float
    block_size: int
    num_blocks: int
    signal_length: int
    primary_rms: float
    reference_rms: float
    estimated_noise_rms: float
    cleaned_speech_rms: float
    nlms_snr_estimate_db: float
    processing_time_sec: float
    real_time_factor: float
    has_nan: bool = False
    has_inf: bool = False
    reference_quality: Optional[ReferenceQualityResult] = None

    @property
    def audio_duration_sec(self) -> float:
        return self.signal_length / self.sample_rate


class NLMSFilter:
    """Normalized LMS filter with safety gating for reference quality."""

    def __init__(self, config: NLMSConfig, reference_quality_config: Optional[ReferenceQualityConfig] = None):
        self.config = config
        self.reference_quality_config = reference_quality_config or ReferenceQualityConfig()
        self._weights: Optional[np.ndarray] = None
        self._last_reference_quality: Optional[ReferenceQualityResult] = None
        self._reset_weights()

    def _reset_weights(self) -> None:
        self._weights = np.zeros(self.config.filter_length, dtype=np.float64)

    def reset(self) -> None:
        self._reset_weights()
        self._last_reference_quality = None

    @property
    def last_reference_quality(self) -> Optional[ReferenceQualityResult]:
        return self._last_reference_quality

    def _process_block(self, x_block, d_block, x_buffer, mu_scale=1.0):
        N = len(x_block)
        y_block = np.zeros(N, dtype=np.float64)
        e_block = np.zeros(N, dtype=np.float64)
        eps = self.config.epsilon
        w = self._weights
        mu = self.config.step_size * float(np.clip(mu_scale, 0.0, 1.0))

        for n in range(N):
            x_buffer = np.roll(x_buffer, 1)
            x_buffer[0] = x_block[n]
            y = float(np.dot(w, x_buffer))
            e = float(d_block[n]) - y
            norm_sq = float(np.dot(x_buffer, x_buffer))
            if mu > 0.0:
                w = w + mu * e * x_buffer / (norm_sq + eps)
            y_block[n] = y
            e_block[n] = e

        self._weights = w
        return y_block, e_block

    def process(self, primary: np.ndarray, reference: np.ndarray, sample_rate: int) -> NLMSResult:
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"sample_rate ({sample_rate}) does not match config.sample_rate ({self.config.sample_rate})"
            )

        primary = np.asarray(primary, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()
        if len(primary) != len(reference):
            raise ValueError(
                f"primary length ({len(primary)}) != reference length ({len(reference)}). "
                "Trim to equal length before calling process()."
            )

        signal_length = len(primary)
        block_size = self.config.block_size
        num_blocks = int(np.ceil(signal_length / block_size)) if signal_length else 0
        estimated_noise = np.zeros(signal_length, dtype=np.float64)
        cleaned_speech = np.zeros(signal_length, dtype=np.float64)
        x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
        self.reset()

        quality = estimate_reference_quality(primary, reference, self.reference_quality_config)
        self._last_reference_quality = quality

        t_start = time.perf_counter()
        for k in range(num_blocks):
            idx_start = k * block_size
            idx_end = min((k + 1) * block_size, signal_length)
            y_block, e_block = self._process_block(
                reference[idx_start:idx_end],
                primary[idx_start:idx_end],
                x_buffer,
                quality.mu_scale,
            )
            hist_start = max(0, idx_end - self.config.filter_length)
            hist = reference[hist_start:idx_end]
            x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
            x_buffer[:len(hist)] = hist[::-1]
            estimated_noise[idx_start:idx_end] = y_block
            cleaned_speech[idx_start:idx_end] = e_block

        processing_time = time.perf_counter() - t_start
        audio_duration = signal_length / sample_rate
        rtf = processing_time / max(audio_duration, 1e-9)
        primary_rms = float(np.sqrt(np.mean(primary ** 2))) if signal_length else 0.0
        reference_rms = float(np.sqrt(np.mean(reference ** 2))) if signal_length else 0.0
        est_noise_rms = float(np.sqrt(np.mean(estimated_noise ** 2))) if signal_length else 0.0
        cleaned_rms = float(np.sqrt(np.mean(cleaned_speech ** 2))) if signal_length else 0.0
        signal_power = float(np.mean(cleaned_speech ** 2)) if signal_length else 0.0
        noise_power = float(np.mean(estimated_noise ** 2)) if signal_length else 0.0
        # This is a diagnostic/proxy SNR, not ground-truth SNR.  Keep it finite
        # even when the adaptive filter estimates effectively zero noise.
        snr_floor = max(float(self.config.epsilon), np.finfo(np.float64).tiny)
        signal_floor = max(signal_power, snr_floor)
        noise_floor = max(noise_power, snr_floor)
        nlms_snr_db = float(10.0 * np.log10(signal_floor / noise_floor))
        if not np.isfinite(nlms_snr_db):
            nlms_snr_db = 0.0
        has_nan = bool(np.any(np.isnan(cleaned_speech)) or np.any(np.isnan(estimated_noise)))
        has_inf = bool(np.any(np.isinf(cleaned_speech)) or np.any(np.isinf(estimated_noise)))

        return NLMSResult(
            cleaned_speech=cleaned_speech.astype(np.float32),
            estimated_noise=estimated_noise.astype(np.float32),
            sample_rate=sample_rate,
            filter_length=self.config.filter_length,
            step_size=self.config.step_size,
            block_size=block_size,
            num_blocks=num_blocks,
            signal_length=signal_length,
            primary_rms=primary_rms,
            reference_rms=reference_rms,
            estimated_noise_rms=est_noise_rms,
            cleaned_speech_rms=cleaned_rms,
            nlms_snr_estimate_db=nlms_snr_db,
            processing_time_sec=processing_time,
            real_time_factor=rtf,
            has_nan=has_nan,
            has_inf=has_inf,
            reference_quality=quality,
        )

    def process_blocks_iter(self, primary: np.ndarray, reference: np.ndarray):
        primary = np.asarray(primary, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()
        if len(primary) != len(reference):
            raise ValueError("primary and reference must have equal length")
        signal_length = len(primary)
        block_size = self.config.block_size
        num_blocks = int(np.ceil(signal_length / block_size)) if signal_length else 0
        x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
        self.reset()
        quality = estimate_reference_quality(primary, reference, self.reference_quality_config)
        self._last_reference_quality = quality
        for k in range(num_blocks):
            idx_start = k * block_size
            idx_end = min((k + 1) * block_size, signal_length)
            y_block, e_block = self._process_block(
                reference[idx_start:idx_end], primary[idx_start:idx_end], x_buffer, quality.mu_scale
            )
            hist_start = max(0, idx_end - self.config.filter_length)
            hist = reference[hist_start:idx_end]
            x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
            x_buffer[:len(hist)] = hist[::-1]
            yield y_block.astype(np.float32), e_block.astype(np.float32)
