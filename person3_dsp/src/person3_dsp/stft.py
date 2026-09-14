"""
stft.py
=======

Owner: Person 3.

Forward transform: real signal -> full complex spectrum D(k, t).

    D(k, t) = sum_{n=0}^{N-1} x[tH + n] * w[n] * exp(-j*2*pi*k*n/N)

Implemented via framing (framing.py) + windowing (windowing.py) +
``numpy.fft.fft`` (a FULL complex FFT -- never ``rfft``, per the
"do not silently switch to rFFT" requirement).

Array convention
-----------------
``D.shape == (num_frames, n_fft)``, i.e. ``(frames, frequency_bins)``.
For Phase 1 (16 kHz): ``D.shape == (num_frames, 320)``.
dtype is complex64 when ``config.dtype == "float32"`` (the default),
else complex128.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DSPConfig
from .framing import frame_signal
from .windowing import get_analysis_window


def stft(
    signal: np.ndarray,
    config: DSPConfig,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the full-complex Short-Time Fourier Transform.

    Parameters
    ----------
    signal:
        1D real-valued mono waveform.
    config:
        Validated :class:`DSPConfig`.
    window:
        Optional precomputed analysis window of shape
        ``(config.window_length,)``. If omitted, a periodic Hann
        window is generated from ``config``. Passing a precomputed
        window avoids recomputation when calling repeatedly (e.g. from
        :class:`~person3_dsp.pipeline.STFTProcessor`).

    Returns
    -------
    np.ndarray
        Complex spectrum ``D`` of shape ``(num_frames, config.n_fft)``.
        This is a FULL complex spectrum: bin ``k`` corresponds to
        frequency ``k * sample_rate / n_fft`` for ``k < n_fft`` using
        standard (non-fftshifted) FFT bin ordering -- bins above the
        Nyquist bin represent negative frequencies, per NumPy's
        ``fft`` convention. No bins are discarded and no rFFT
        conversion is performed.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D (mono), got shape {signal.shape}")

    np_dtype = np.float32 if config.dtype == "float32" else np.float64
    complex_dtype = np.complex64 if config.dtype == "float32" else np.complex128

    if window is None:
        window = get_analysis_window(config.window_type, config.window_length, dtype=np_dtype)
    elif window.shape[0] != config.window_length:
        raise ValueError(
            f"window has length {window.shape[0]}, expected "
            f"config.window_length={config.window_length}"
        )

    frames = frame_signal(signal.astype(np_dtype, copy=False), config)  # (nf, window_length)
    windowed = frames * window[np.newaxis, :]

    if config.n_fft > config.window_length:
        pad_width = config.n_fft - config.window_length
        windowed = np.pad(windowed, ((0, 0), (0, pad_width)))

    D = np.fft.fft(windowed, n=config.n_fft, axis=1)
    return D.astype(complex_dtype)
