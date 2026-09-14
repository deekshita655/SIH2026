"""
istft.py
========

Owner: Person 3.

Inverse transform: complex spectrum S(k, t) -> real time-domain signal
d_AI[n], via IFFT -> synthesis window -> overlap-add -> explicit
normalization (ola.py).

This module never concatenates IFFT frames directly -- that would
produce clicking/discontinuities at every hop boundary. It always
routes through :func:`person3_dsp.ola.overlap_add`, which divides out
the accumulated window energy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DSPConfig
from .windowing import get_analysis_window
from .ola import overlap_add


def istft(
    S: np.ndarray,
    config: DSPConfig,
    output_length: Optional[int] = None,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse STFT: full complex spectrum -> real time-domain signal.

    Parameters
    ----------
    S:
        Complex spectrum, shape ``(num_frames, config.n_fft)``. This
        may be the original ``D`` (round-trip test) or an
        AI-enhanced spectrum with the same shape/dtype convention.
    config:
        The SAME :class:`DSPConfig` used for the forward STFT that
        produced (or is shape-compatible with) ``S``.
    output_length:
        If given, the reconstructed signal is truncated/zero-padded to
        exactly this many samples (typically the original input
        signal's length, to strip any tail zero-padding introduced by
        ``tail_policy="zero_pad"`` during the forward pass). If
        ``None``, the natural overlap-add length is returned:
        ``(num_frames - 1) * hop_length + window_length``.
    window:
        Optional precomputed window of shape ``(config.window_length,)``,
        used for BOTH analysis-side normalization and synthesis. If
        omitted, a periodic Hann window is generated from ``config``.
        Using the same window for analysis and synthesis (as required)
        is handled automatically here.

    Returns
    -------
    np.ndarray
        Real-valued reconstructed signal, dtype matching
        ``config.dtype``.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"S must be 2D (num_frames, n_fft), got shape {S.shape}")
    if S.shape[1] != config.n_fft:
        raise ValueError(
            f"S has {S.shape[1]} frequency bins, expected config.n_fft={config.n_fft}"
        )

    np_dtype = np.float32 if config.dtype == "float32" else np.float64

    if window is None:
        window = get_analysis_window(config.window_type, config.window_length, dtype=np_dtype)
    elif window.shape[0] != config.window_length:
        raise ValueError(
            f"window has length {window.shape[0]}, expected "
            f"config.window_length={config.window_length}"
        )

    # IFFT of the full complex spectrum, per frame.
    time_frames_complex = np.fft.ifft(S, n=config.n_fft, axis=1)

    # The IFFT of a spectrum derived from a real windowed frame is (up
    # to floating point error) real-valued. We take the real part
    # explicitly rather than silently discarding phase earlier -- the
    # complex spectrum itself was never truncated; this is standard
    # ISTFT reconstruction of a real-valued waveform.
    time_frames = np.real(time_frames_complex).astype(np_dtype)

    if config.n_fft > config.window_length:
        # Only the first window_length samples correspond to the
        # windowed-analysis support; see README "known limitations"
        # for the n_fft > window_length case.
        time_frames = time_frames[:, : config.window_length]

    output, _window_accum = overlap_add(
        frames_time=time_frames,
        hop_length=config.hop_length,
        synthesis_window=window,
        analysis_window=window,
        output_length=output_length,
    )
    return output.astype(np_dtype)
