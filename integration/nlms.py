"""
nlms.py
=======
Person 2 — NLMS Adaptive Noise Cancellation
Python development implementation following the Person 2 MATLAB
NLMS configuration/reference.

IMPORTANT: This is a Python port of the Person 2 MATLAB reference:

    dsp.LMSFilter('Method','Normalized LMS','Length',64,'StepSize',0.01)

It does NOT claim to literally execute MATLAB's dsp.LMSFilter. The
algorithm is mathematically equivalent for the same parameters.

The Normalized LMS update rule:
    y[n]   = w^T x[n]               (filter output = estimated noise)
    e[n]   = d[n] - y[n]            (error = cleaned speech)
    w[n+1] = w[n] + μ e[n] x[n] / (||x[n]||² + ε)

where:
    x[n] = reference signal buffer (length = filter_length)
    d[n] = primary signal (speech + noise)
    y[n] = estimated correlated noise component
    e[n] = cleaned speech (error signal)
    w    = adaptive filter weights (length = filter_length)
    μ    = step size
    ε    = numerical stability term

NLMS relationship to the pipeline:
    Primary microphone  = speech + noise  (d[n])
    Reference microphone= noise reference (x[n])
    The NLMS filter adaptively estimates the correlated noise in the
    primary signal from the reference, producing:
        y[n] = estimated noise
        e[n] = cleaned speech

Design follows the Person 2 MATLAB script structure:
    1. Load audio (handled externally)
    2. Verify sampling rates (handled externally)
    3. Convert to mono (handled externally or by NLMSFilter)
    4. Match lengths (handled externally)
    5. Initialize filter weights to zero
    6. Process in blocks of block_size samples
    7. Return cleaned speech, estimated noise, diagnostics

Reference MATLAB defaults:
    Fs_expected  = 16000
    filterLength = 64
    stepSize     = 0.01
    blockSize    = 1024
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import NLMSConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NLMSResult:
    """
    Output from one NLMS processing run.

    All SNR values are ESTIMATES, not ground truth.
    The NLMS-derived SNR is a reference-based estimate using
    primary power / estimated_noise power — it is NOT identical to
    the dataset's target SNR.
    """

    # Audio outputs
    cleaned_speech: np.ndarray          # e[n] = primary - estimated_noise
    estimated_noise: np.ndarray         # y[n] = NLMS filter output

    # Configuration echo
    sample_rate: int
    filter_length: int
    step_size: float
    block_size: int
    num_blocks: int
    signal_length: int                  # samples

    # Amplitude metrics (RMS)
    primary_rms: float                  # RMS of original primary signal
    reference_rms: float                # RMS of reference signal
    estimated_noise_rms: float          # RMS of NLMS estimated noise y[n]
    cleaned_speech_rms: float           # RMS of cleaned speech e[n]

    # NLMS-derived SNR estimate
    # IMPORTANT: This is an ESTIMATE derived from signal powers.
    # It is NOT the dataset ground-truth SNR.
    nlms_snr_estimate_db: float

    # Performance
    processing_time_sec: float
    real_time_factor: float             # processing_time / audio_duration

    # Validity flags
    has_nan: bool = False
    has_inf: bool = False

    @property
    def audio_duration_sec(self) -> float:
        return self.signal_length / self.sample_rate


# ---------------------------------------------------------------------------
# NLMS Filter
# ---------------------------------------------------------------------------

class NLMSFilter:
    """
    Normalized Least Mean Squares adaptive filter.

    Python development implementation following the Person 2 MATLAB
    NLMS configuration/reference (dsp.LMSFilter 'Normalized LMS').

    This class processes audio in blocks matching the Person 2 MATLAB
    reference (default block_size=1024).

    Args:
        config: NLMSConfig — all algorithm parameters

    Usage:
        cfg = NLMSConfig(filter_length=64, step_size=0.01, block_size=1024)
        filt = NLMSFilter(cfg)
        result = filt.process(primary, reference, sample_rate=16000)
    """

    def __init__(self, config: NLMSConfig):
        self.config = config
        self._weights: Optional[np.ndarray] = None
        self._reset_weights()

    def _reset_weights(self) -> None:
        """Initialise filter weights to zero (matches MATLAB dsp.LMSFilter initial state)."""
        self._weights = np.zeros(self.config.filter_length, dtype=np.float64)

    def reset(self) -> None:
        """Reset filter state. Call between independent audio segments."""
        self._reset_weights()

    # ------------------------------------------------------------------
    # Core block-level NLMS step
    # ------------------------------------------------------------------

    def _process_block(
        self,
        x_block: np.ndarray,   # reference signal block
        d_block: np.ndarray,   # primary signal block
        x_buffer: np.ndarray,  # circular history of reference samples
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process one block of samples using NLMS.

        Implements the MATLAB-equivalent per-sample NLMS update loop:
            y[n]   = w^T x[n]
            e[n]   = d[n] - y[n]
            w[n+1] = w[n] + μ * e[n] * x[n] / (||x[n]||² + ε)

        Returns:
            y_block: estimated noise (filter output)
            e_block: cleaned speech (error signal)
        """
        N = len(x_block)
        y_block = np.zeros(N, dtype=np.float64)
        e_block = np.zeros(N, dtype=np.float64)
        L = self.config.filter_length
        mu = self.config.step_size
        eps = self.config.epsilon
        w = self._weights

        for n in range(N):
            # Shift reference sample into buffer (newest first)
            x_buffer = np.roll(x_buffer, 1)
            x_buffer[0] = x_block[n]

            # Filter output: y[n] = w^T x[n]
            y = float(np.dot(w, x_buffer))

            # Error: e[n] = d[n] - y[n]
            e = float(d_block[n]) - y

            # Normalised LMS weight update
            norm_sq = float(np.dot(x_buffer, x_buffer))
            w = w + mu * e * x_buffer / (norm_sq + eps)

            y_block[n] = y
            e_block[n] = e

        self._weights = w
        return y_block, e_block

    # ------------------------------------------------------------------
    # Public processing entry points
    # ------------------------------------------------------------------

    def process(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> NLMSResult:
        """
        Full offline NLMS processing, mirroring the Person 2 MATLAB script.

        Processes primary and reference signals block-by-block
        (block_size = config.block_size, default 1024 — matches MATLAB).

        Args:
            primary:    speech + noise signal (1D float, normalised ~[-1,1])
            reference:  reference noise signal (1D float, same length)
            sample_rate: sampling rate in Hz (must match config.sample_rate)

        Returns:
            NLMSResult with cleaned speech, estimated noise, and diagnostics
        """
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"sample_rate ({sample_rate}) does not match "
                f"config.sample_rate ({self.config.sample_rate})"
            )

        # Ensure 1D float64 for numerical precision during NLMS
        primary = np.asarray(primary, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()

        if len(primary) != len(reference):
            raise ValueError(
                f"primary length ({len(primary)}) != reference length ({len(reference)}). "
                "Trim to equal length before calling process()."
            )

        signal_length = len(primary)
        block_size = self.config.block_size
        num_blocks = int(np.ceil(signal_length / block_size))

        estimated_noise = np.zeros(signal_length, dtype=np.float64)
        cleaned_speech = np.zeros(signal_length, dtype=np.float64)

        # Reference sample buffer (newest-first order for inner product)
        x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)

        self.reset()  # ensure clean start

        t_start = time.perf_counter()

        for k in range(num_blocks):
            idx_start = k * block_size
            idx_end = min((k + 1) * block_size, signal_length)

            x_block = reference[idx_start:idx_end]
            d_block = primary[idx_start:idx_end]

            y_block, e_block = self._process_block(x_block, d_block, x_buffer)

            # Update x_buffer state for next block
            # (the inner loop already updated it via roll; we need the last state)
            # Re-derive from the last filter_length samples of reference up to this block
            hist_start = max(0, idx_end - self.config.filter_length)
            hist = reference[hist_start:idx_end]
            x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
            x_buffer[:len(hist)] = hist[::-1]  # newest first

            estimated_noise[idx_start:idx_end] = y_block
            cleaned_speech[idx_start:idx_end] = e_block

        t_end = time.perf_counter()
        processing_time = t_end - t_start
        audio_duration = signal_length / sample_rate
        rtf = processing_time / max(audio_duration, 1e-9)

        # Metrics
        primary_rms = float(np.sqrt(np.mean(primary ** 2)))
        reference_rms = float(np.sqrt(np.mean(reference ** 2)))
        est_noise_rms = float(np.sqrt(np.mean(estimated_noise ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned_speech ** 2)))

        # NLMS-derived estimated SNR (reference-based estimate, NOT ground truth)
        signal_power = float(np.mean(cleaned_speech ** 2))
        noise_power = float(np.mean(estimated_noise ** 2))
        if noise_power > 1e-12:
            nlms_snr_db = 10.0 * np.log10(signal_power / noise_power)
        else:
            nlms_snr_db = float("inf")

        # Validity checks
        has_nan = bool(np.any(np.isnan(cleaned_speech)) or np.any(np.isnan(estimated_noise)))
        has_inf = bool(np.any(np.isinf(cleaned_speech)) or np.any(np.isinf(estimated_noise)))

        return NLMSResult(
            cleaned_speech=cleaned_speech.astype(np.float32),
            estimated_noise=estimated_noise.astype(np.float32),
            sample_rate=sample_rate,
            filter_length=self.config.filter_length,
            step_size=self.config.step_size,
            block_size=self.config.block_size,
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
        )

    def process_blocks_iter(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
    ):
        """
        Generator: yield (y_block, e_block) for each block in order.

        Useful for streaming integration where each block feeds into P3's
        StreamingFramer without buffering the entire output.

        Yields:
            (y_block, e_block): estimated noise and cleaned speech for this block,
                both as float32 arrays of length <= config.block_size
        """
        primary = np.asarray(primary, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()

        signal_length = len(primary)
        block_size = self.config.block_size
        num_blocks = int(np.ceil(signal_length / block_size))

        x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
        self.reset()

        for k in range(num_blocks):
            idx_start = k * block_size
            idx_end = min((k + 1) * block_size, signal_length)

            x_block = reference[idx_start:idx_end]
            d_block = primary[idx_start:idx_end]

            y_block, e_block = self._process_block(x_block, d_block, x_buffer)

            # Update buffer from history
            hist_start = max(0, idx_end - self.config.filter_length)
            hist = reference[hist_start:idx_end]
            x_buffer = np.zeros(self.config.filter_length, dtype=np.float64)
            x_buffer[:len(hist)] = hist[::-1]

            yield y_block.astype(np.float32), e_block.astype(np.float32)
