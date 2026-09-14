"""
framing.py
==========

Owner: Person 3.

Explicit, non-centered ("streaming-style") framing.

Convention
----------
Frame ``t`` covers samples::

    x[t * hop_length : t * hop_length + window_length]

There is NO centering (frame 0 is NOT centered on sample 0) and NO
implicit/automatic padding. This matches the requirement to support
real-time streaming, where future samples are not available and
library defaults such as librosa's ``center=True`` (which pads
``n_fft // 2`` samples on each side) cannot be used.

Two usage modes are provided:

1. Offline: :func:`frame_signal` frames an entire in-memory signal at
   once, applying an explicit, documented ``tail_policy`` to whatever
   samples are left over at the end.
2. Streaming: :class:`StreamingFramer` accepts audio incrementally
   (``push``), emits only frames that are fully available, and lets
   the caller decide when to ``flush`` the final partial frame
   (e.g. at end-of-stream / end-of-utterance).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import DSPConfig


def compute_num_frames(
    signal_length: int,
    window_length: int,
    hop_length: int,
    tail_policy: str,
) -> int:
    """Compute how many frames :func:`frame_signal` will produce.

    Parameters
    ----------
    signal_length:
        Number of samples in the signal.
    window_length, hop_length:
        Frame size and hop size in samples.
    tail_policy:
        ``"drop"`` or ``"zero_pad"``.

    Returns
    -------
    int
        Number of frames, following the documented tail policy.
    """
    if tail_policy not in ("drop", "zero_pad"):
        raise ValueError(f"Unsupported tail_policy: {tail_policy!r}")
    if signal_length <= 0:
        return 0

    if signal_length < window_length:
        # Not even one full frame available.
        return 1 if tail_policy == "zero_pad" else 0

    # Number of frames whose window is *fully* contained in the signal.
    n_full = (signal_length - window_length) // hop_length + 1
    next_start = n_full * hop_length

    if tail_policy == "zero_pad" and next_start < signal_length:
        # There are leftover samples after the last full frame that
        # don't fill a complete window -- pad one more frame.
        return n_full + 1
    return n_full


def frame_signal(signal: np.ndarray, config: DSPConfig) -> np.ndarray:
    """Frame a full (offline) 1D signal according to ``config``.

    Applies ``config.tail_policy`` explicitly:

    - ``"drop"``: any trailing samples that don't complete a full
      ``window_length``-sample frame are discarded.
    - ``"zero_pad"``: the final incomplete frame is zero-padded up to
      ``window_length`` and included.

    Parameters
    ----------
    signal:
        1D real-valued array.
    config:
        Validated :class:`DSPConfig`.

    Returns
    -------
    np.ndarray
        Shape ``(num_frames, window_length)``, dtype matching
        ``config.dtype``.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D (mono), got shape {signal.shape}")

    np_dtype = np.float32 if config.dtype == "float32" else np.float64
    n = signal.shape[0]
    num_frames = compute_num_frames(
        n, config.window_length, config.hop_length, config.tail_policy
    )
    frames = np.zeros((num_frames, config.window_length), dtype=np_dtype)
    for t in range(num_frames):
        start = t * config.hop_length
        end = start + config.window_length
        chunk = signal[start:min(end, n)]
        frames[t, : chunk.shape[0]] = chunk
    return frames


def frame_start_samples(num_frames: int, hop_length: int) -> np.ndarray:
    """Sample index at which each frame *starts* (frame-START convention)."""
    return np.arange(num_frames, dtype=np.int64) * hop_length


def frame_start_times(num_frames: int, hop_length: int, sample_rate: int) -> np.ndarray:
    """Time in seconds at which each frame *starts* (frame-START convention).

    These are frame **start** timestamps, not centers. See README
    section "Frame timing" for the rationale (streaming systems know a
    frame's start time as soon as the frame is available; the center
    time only exists in hindsight for the offline case).
    """
    return frame_start_samples(num_frames, hop_length) / float(sample_rate)


def frame_center_times(num_frames: int, hop_length: int, window_length: int, sample_rate: int) -> np.ndarray:
    """Optional: time in seconds at each frame's *center*.

    Provided for convenience/analysis (e.g. plotting against a
    spectrogram) but is NOT the primary timestamp convention used by
    this module -- see :func:`frame_start_times`.
    """
    starts = frame_start_samples(num_frames, hop_length).astype(np.float64)
    return (starts + window_length / 2.0) / float(sample_rate)


class StreamingFramer:
    """Incremental framer for real-time / streaming use.

    Usage
    -----
    ::

        framer = StreamingFramer(config)
        for chunk in audio_chunks:
            frames = framer.push(chunk)   # list of complete frames, may be empty
            for frame in frames:
                ...  # process each complete frame as it becomes available
        last = framer.flush()             # None, or the final (possibly
                                           # zero-padded) partial frame
    """

    def __init__(self, config: DSPConfig):
        self.config = config
        self._np_dtype = np.float32 if config.dtype == "float32" else np.float64
        self._buffer = np.zeros(0, dtype=self._np_dtype)

    def push(self, samples: np.ndarray) -> List[np.ndarray]:
        """Feed new samples in; return the list of newly-complete frames.

        Frames are emitted in order, each of shape ``(window_length,)``.
        Internally the buffer only retains what is still needed for
        future frames (i.e. it advances by ``hop_length`` per emitted
        frame), so memory use does not grow unbounded during a long
        stream.
        """
        samples = np.asarray(samples, dtype=self._np_dtype).reshape(-1)
        self._buffer = np.concatenate([self._buffer, samples])

        window_length = self.config.window_length
        hop_length = self.config.hop_length
        frames: List[np.ndarray] = []
        while self._buffer.shape[0] >= window_length:
            frames.append(self._buffer[:window_length].copy())
            self._buffer = self._buffer[hop_length:]
        return frames

    def flush(self) -> Optional[np.ndarray]:
        """Finalize the stream and return the trailing partial frame.

        Applies ``config.tail_policy``:

        - ``"drop"``: returns ``None`` and discards the leftover
          samples.
        - ``"zero_pad"``: zero-pads the leftover samples up to
          ``window_length`` and returns that frame. Returns ``None``
          if there were no leftover samples at all.

        After calling ``flush``, the internal buffer is cleared; the
        framer can be reused for a new stream.
        """
        try:
            if self._buffer.shape[0] == 0:
                return None
            if self.config.tail_policy == "drop":
                return None
            window_length = self.config.window_length
            pad = np.zeros(window_length - self._buffer.shape[0], dtype=self._np_dtype)
            frame = np.concatenate([self._buffer, pad])
            return frame
        finally:
            self._buffer = np.zeros(0, dtype=self._np_dtype)

    def reset(self) -> None:
        """Discard any buffered samples without applying the tail policy."""
        self._buffer = np.zeros(0, dtype=self._np_dtype)

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently buffered but not yet emitted."""
        return int(self._buffer.shape[0])
