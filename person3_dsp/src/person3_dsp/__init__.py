"""
person3_dsp
===========

Person 3's Time-Frequency DSP module: STFT / ISTFT / overlap-add for
the speech enhancement project.

Public API
----------
- ``DSPConfig``: explicit, validated configuration (sample rate,
  window/hop size, FFT size, tail policy, ...).
- ``STFTProcessor``: high-level forward/inverse API
  (``transform`` / ``inverse``) plus frame-timing and frequency-bin
  helpers. This is what other team members should use.
- ``spectrum``: submodule with magnitude/phase/real/imag/reconstruct
  helpers for consuming a spectrum without touching FFT internals.

Lower-level building blocks (``stft``, ``istft``, ``framing``,
``windowing``, ``ola``) are also importable directly for advanced use
or testing, but ``STFTProcessor`` is the intended entry point.
"""

from .config import DSPConfig
from .pipeline import STFTProcessor
from . import spectrum

__all__ = [
    "DSPConfig",
    "STFTProcessor",
    "spectrum",
]

__version__ = "0.1.0"
