"""
pipeline.py
===========

Owner: Person 3.

Public, high-level API: :class:`STFTProcessor`.

This is the class other team members (P4, P5) should use; it hides
the internal framing / windowing / FFT details behind
``transform`` / ``inverse`` plus small helpers for frame timing and
frequency bins.

Example
-------
::

    from person3_dsp import DSPConfig, STFTProcessor

    config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(config)

    D = dsp.transform(signal)              # (num_frames, 320) complex64
    # ... S = model(D) elsewhere (owned by Person 4) ...
    reconstructed = dsp.inverse(D, output_length=len(signal))
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DSPConfig
from .windowing import get_analysis_window
from .framing import (
    StreamingFramer,
    compute_num_frames,
    frame_start_samples,
    frame_start_times,
    frame_center_times,
)
from .stft import stft
from .istft import istft
from . import spectrum as _spectrum


class STFTProcessor:
    """Clean forward/inverse STFT API built on top of ``DSPConfig``.

    Owns exactly one precomputed analysis/synthesis window (periodic
    Hann by default), reused across calls for efficiency and to
    guarantee analysis and synthesis use the identical window.
    """

    def __init__(self, config: DSPConfig):
        self.config = config
        self._np_dtype = np.float32 if config.dtype == "float32" else np.float64
        self._window = get_analysis_window(
            config.window_type, config.window_length, dtype=self._np_dtype
        )

    # ------------------------------------------------------------------
    # Forward / inverse
    # ------------------------------------------------------------------
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Forward STFT. ``signal`` (1D real) -> ``D`` (complex, (frames, n_fft))."""
        return stft(signal, self.config, window=self._window)

    def inverse(self, S: np.ndarray, output_length: Optional[int] = None) -> np.ndarray:
        """Inverse STFT. ``S`` (complex, (frames, n_fft)) -> real signal.

        Pass ``output_length=len(original_signal)`` to strip any
        tail zero-padding that was added during the forward transform.
        """
        return istft(S, self.config, output_length=output_length, window=self._window)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def new_streaming_framer(self) -> StreamingFramer:
        """Create a fresh :class:`StreamingFramer` bound to this config."""
        return StreamingFramer(self.config)

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """Forward STFT of a SINGLE already-framed block of samples.

        ``frame`` must have shape ``(config.window_length,)`` (e.g. as
        produced by :class:`StreamingFramer`). Returns a complex
        vector of shape ``(config.n_fft,)``.
        """
        frame = np.asarray(frame)
        if frame.shape != (self.config.window_length,):
            raise ValueError(
                f"frame must have shape ({self.config.window_length},), "
                f"got {frame.shape}"
            )
        D = stft(frame, self.config, window=self._window)  # shape (1, n_fft)
        return D[0]

    # ------------------------------------------------------------------
    # Metadata helpers (frame timing, frequency bins, num frames)
    # ------------------------------------------------------------------
    def num_frames_for_length(self, signal_length: int) -> int:
        """Number of frames :meth:`transform` will produce for a signal
        of this many samples (given ``config.tail_policy``)."""
        return compute_num_frames(
            signal_length, self.config.window_length, self.config.hop_length,
            self.config.tail_policy,
        )

    def frame_start_samples(self, num_frames: int) -> np.ndarray:
        """Sample index at which each frame starts (frame-START convention)."""
        return frame_start_samples(num_frames, self.config.hop_length)

    def frame_times(self, num_frames: int) -> np.ndarray:
        """Time (seconds) at which each frame STARTS. Primary timestamp
        convention -- see README 'Frame timing'."""
        return frame_start_times(num_frames, self.config.hop_length, self.config.sample_rate)

    def frame_center_times(self, num_frames: int) -> np.ndarray:
        """Optional: time (seconds) at each frame's CENTER. Not the
        primary streaming convention; see :meth:`frame_times`."""
        return frame_center_times(
            num_frames, self.config.hop_length, self.config.window_length, self.config.sample_rate
        )

    def frequency_bins(self) -> np.ndarray:
        """Frequency (Hz) represented by each of the ``config.n_fft`` bins."""
        return _spectrum.frequency_bins(self.config.sample_rate, self.config.n_fft)

    @property
    def window(self) -> np.ndarray:
        """The (periodic Hann, by default) analysis/synthesis window in use."""
        return self._window
