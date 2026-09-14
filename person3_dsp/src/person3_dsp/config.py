"""
config.py
=========

Owner: Person 3 (Time-Frequency DSP module).

Defines the single source of truth for STFT/ISTFT configuration:
``DSPConfig``.

Design goals
------------
- Nothing in the rest of the package hard-codes 320 / 160 (the Phase-1
  16 kHz values). Every derived quantity is computed from
  ``sample_rate``, ``window_ms`` and ``hop_ms`` so the exact same code
  works for the Phase-1 (16 kHz) and future production (48 kHz)
  configurations.
- All important behavioral choices (windowing, centering, tail policy)
  are explicit fields on this object rather than implicit library
  defaults, per the project's "do not cheat" requirements.

Derivation
----------
::

    window_length = round(sample_rate * window_ms / 1000)
    hop_length    = round(sample_rate * hop_ms / 1000)
    n_fft         = window_length      (unless explicitly overridden)

16 kHz (Phase 1):  window_length=320, hop_length=160, n_fft=320
48 kHz (Production): window_length=960, hop_length=480, n_fft=960
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Only "hann" is implemented in Phase 1. Kept as a set (not a Literal)
# so validation gives a clear runtime error message.
_SUPPORTED_WINDOW_TYPES = {"hann"}
_SUPPORTED_TAIL_POLICIES = {"drop", "zero_pad"}
_SUPPORTED_DTYPES = {"float32", "float64"}


@dataclass
class DSPConfig:
    """Explicit, validated configuration for the STFT/ISTFT pipeline.

    Parameters
    ----------
    sample_rate:
        Sampling rate in Hz. Must be a positive integer. Phase 1 uses
        16000; the production target is 48000. The code makes no
        assumption about which value is used.
    window_ms:
        Analysis window duration in milliseconds. Default 20 ms.
    hop_ms:
        Hop (frame advance) duration in milliseconds. Default 10 ms
        (50% overlap when window_ms=20).
    n_fft:
        FFT size. If ``None`` (default), it is set equal to
        ``window_length`` (no zero-padding beyond the analysis
        window). If explicitly provided it must be >= window_length.
    window_type:
        Name of the analysis/synthesis window. Only ``"hann"`` is
        supported in Phase 1.
    center:
        Must be ``False``. This module implements only non-centered,
        streaming-compatible framing (frame t starts at
        ``t * hop_length``). ``center=True`` (librosa-style padding
        that centers frame 0 on sample 0) is explicitly NOT supported,
        so that streaming and offline processing behave identically.
    tail_policy:
        How to handle the final, possibly-incomplete frame of an
        offline signal:

        - ``"drop"``: discard any samples that do not fill a
          complete ``window_length``-sample frame.
        - ``"zero_pad"``: zero-pad the final partial frame up to
          ``window_length`` and process it as one more frame.

        Default is ``"zero_pad"`` (no audio is silently discarded).
    dtype:
        Internal floating point dtype used for time-domain DSP
        (``"float32"`` or ``"float64"``). Complex spectra use the
        matching complex dtype (complex64 / complex128). Default
        ``"float32"``.

    Derived / computed attributes (not passed to the constructor)
    ----------------------------------------------------------------
    window_length:
        ``round(sample_rate * window_ms / 1000)`` samples.
    hop_length:
        ``round(sample_rate * hop_ms / 1000)`` samples.
    """

    sample_rate: int
    window_ms: float = 20.0
    hop_ms: float = 10.0
    n_fft: Optional[int] = None
    window_type: str = "hann"
    center: bool = False
    tail_policy: str = "zero_pad"
    dtype: str = "float32"

    # Derived fields, computed in __post_init__. Declared with
    # init=False so callers cannot pass inconsistent values directly.
    window_length: int = field(init=False)
    hop_length: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_rate, (int,)) or self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive integer, got {self.sample_rate!r}"
            )
        if self.window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {self.window_ms!r}")
        if self.hop_ms <= 0:
            raise ValueError(f"hop_ms must be positive, got {self.hop_ms!r}")
        if self.center:
            raise ValueError(
                "center=True is not supported. This module implements only "
                "non-centered (streaming-style) framing: frame t covers "
                "signal[t*hop_length : t*hop_length + window_length]. "
                "Centered/padded STFT is out of scope by design (see README "
                "section 'Non-centered streaming behavior')."
            )
        if self.window_type not in _SUPPORTED_WINDOW_TYPES:
            raise ValueError(
                f"Unsupported window_type {self.window_type!r}; "
                f"supported types: {sorted(_SUPPORTED_WINDOW_TYPES)}"
            )
        if self.tail_policy not in _SUPPORTED_TAIL_POLICIES:
            raise ValueError(
                f"Unsupported tail_policy {self.tail_policy!r}; "
                f"supported policies: {sorted(_SUPPORTED_TAIL_POLICIES)}"
            )
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}; "
                f"supported dtypes: {sorted(_SUPPORTED_DTYPES)}"
            )

        # --- derive window_length / hop_length explicitly from ms ---
        window_length = round(self.sample_rate * self.window_ms / 1000.0)
        hop_length = round(self.sample_rate * self.hop_ms / 1000.0)

        if window_length <= 0:
            raise ValueError(
                f"Derived window_length must be positive, got {window_length} "
                f"(sample_rate={self.sample_rate}, window_ms={self.window_ms})"
            )
        if hop_length <= 0:
            raise ValueError(
                f"Derived hop_length must be positive, got {hop_length} "
                f"(sample_rate={self.sample_rate}, hop_ms={self.hop_ms})"
            )
        if hop_length > window_length:
            raise ValueError(
                f"hop_length ({hop_length}) must not exceed window_length "
                f"({window_length}); this would leave gaps between frames "
                "that overlap-add normalization cannot recover."
            )

        self.window_length = window_length
        self.hop_length = hop_length

        if self.n_fft is None:
            self.n_fft = window_length
        if self.n_fft < window_length:
            raise ValueError(
                f"n_fft ({self.n_fft}) must be >= window_length "
                f"({window_length})."
            )

    @property
    def complex_dtype(self) -> str:
        """Complex dtype matching ``self.dtype`` (complex64/complex128)."""
        return "complex64" if self.dtype == "float32" else "complex128"

    @property
    def overlap_fraction(self) -> float:
        """Fractional overlap between consecutive frames, in [0, 1)."""
        return 1.0 - (self.hop_length / self.window_length)

    def frequency_resolution_hz(self) -> float:
        """Spacing between adjacent FFT bins in Hz: ``sample_rate / n_fft``."""
        return self.sample_rate / self.n_fft

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "DSPConfig("
            f"sample_rate={self.sample_rate}, "
            f"window_ms={self.window_ms}, hop_ms={self.hop_ms}, "
            f"window_length={self.window_length}, hop_length={self.hop_length}, "
            f"n_fft={self.n_fft}, window_type={self.window_type!r}, "
            f"center={self.center}, tail_policy={self.tail_policy!r}, "
            f"dtype={self.dtype!r})"
        )
