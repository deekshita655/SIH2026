"""
ola.py
======

Owner: Person 3.

Generic, standalone overlap-add (OLA) with EXPLICIT window
normalization. This is deliberately independent of the FFT / STFT
machinery so it can be unit-tested on its own (see
``tests/test_ola.py``) and reused by ``istft.py``.

Overlap-add is never a bare summation of frames -- concatenating or
naively summing windowed frames without dividing out the accumulated
window energy produces amplitude modulation. This module always
returns (and callers always divide by) the window accumulator.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def overlap_add(
    frames_time: np.ndarray,
    hop_length: int,
    synthesis_window: np.ndarray,
    analysis_window: Optional[np.ndarray] = None,
    output_length: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-add a stack of time-domain frames with normalization.

    Conceptually::

        signal_accumulator[start:start+N] += frame_time_domain * synthesis_window
        window_accumulator[start:start+N] += analysis_window * synthesis_window
        output = signal_accumulator / window_accumulator   (window_accumulator kept >= eps)

    Parameters
    ----------
    frames_time:
        Real-valued array of shape ``(num_frames, frame_length)``.
    hop_length:
        Hop size in samples between consecutive frame starts.
    synthesis_window:
        Shape ``(frame_length,)``, applied to each frame before
        accumulation.
    analysis_window:
        Shape ``(frame_length,)``, the window that was applied during
        the forward/analysis stage. Used only to build the
        normalization denominator (``analysis_window * synthesis_window``).
        If ``None``, an all-ones array is used (i.e. it is assumed no
        analysis window was applied upstream).
    output_length:
        If given, the output is truncated or zero-padded to exactly
        this many samples. If ``None`` (default), the natural length
        ``(num_frames - 1) * hop_length + frame_length`` is used.
    eps:
        Floor applied to the window accumulator before dividing, so we
        NEVER divide by exactly zero. Positions where the true
        window-accumulator value is smaller than ``eps`` are replaced
        by ``eps`` (not by ``1.0`` and not skipped), so the boundary
        behavior stays deterministic and documented.

    Returns
    -------
    (output, window_accumulator):
        ``output`` is the normalized, reconstructed real signal
        (shape ``(output_length,)``). ``window_accumulator`` is
        returned for diagnostics/tests (e.g. to verify COLA behavior
        or inspect boundary tapering) and has the natural
        (non-truncated) length.
    """
    frames_time = np.asarray(frames_time)
    if frames_time.ndim != 2:
        raise ValueError(f"frames_time must be 2D (num_frames, frame_length), got shape {frames_time.shape}")
    num_frames, frame_length = frames_time.shape

    synthesis_window = np.asarray(synthesis_window)
    if synthesis_window.shape[0] != frame_length:
        raise ValueError(
            f"synthesis_window length {synthesis_window.shape[0]} != "
            f"frame_length {frame_length}"
        )
    if analysis_window is None:
        analysis_window = np.ones(frame_length, dtype=synthesis_window.dtype)
    else:
        analysis_window = np.asarray(analysis_window)
        if analysis_window.shape[0] != frame_length:
            raise ValueError(
                f"analysis_window length {analysis_window.shape[0]} != "
                f"frame_length {frame_length}"
            )
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    work_dtype = np.float64  # accumulate in float64 for numerical stability
    natural_length = (num_frames - 1) * hop_length + frame_length if num_frames > 0 else 0

    signal_accum = np.zeros(natural_length, dtype=work_dtype)
    window_accum = np.zeros(natural_length, dtype=work_dtype)
    combined_window = (analysis_window.astype(work_dtype) * synthesis_window.astype(work_dtype))

    for t in range(num_frames):
        start = t * hop_length
        end = start + frame_length
        signal_accum[start:end] += frames_time[t].astype(work_dtype) * synthesis_window.astype(work_dtype)
        window_accum[start:end] += combined_window

    safe_window = np.where(window_accum > eps, window_accum, eps)
    output = signal_accum / safe_window

    target_length = natural_length if output_length is None else output_length
    if target_length < natural_length:
        output = output[:target_length]
    elif target_length > natural_length:
        output = np.pad(output, (0, target_length - natural_length))

    out_dtype = np.float32 if synthesis_window.dtype == np.float32 else np.float64
    return output.astype(out_dtype), window_accum
