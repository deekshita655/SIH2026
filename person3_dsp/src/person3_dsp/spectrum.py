"""
spectrum.py
===========

Owner: Person 3.

Read-only utilities for consuming the canonical complex spectrum
``D(k, t)`` (or the AI-enhanced spectrum ``S(k, t)``, same shape/dtype
convention) without needing to know FFT internals. This is the primary
surface Person 4 (model layer) and Person 5 (acoustic-intelligence
features) should use.

No phase information is ever discarded by these helpers; the complex
array itself remains the canonical representation. ``reconstruct``
below is provided only as a round-trip check / convenience -- it does
NOT unwrap phase, and callers should not assume unwrapped phase unless
they compute it themselves (this module never unwraps automatically).
"""

from __future__ import annotations

import numpy as np


def magnitude(D: np.ndarray) -> np.ndarray:
    """Magnitude |D(k, t)|. Real-valued, same shape as D."""
    return np.abs(D)


def phase(D: np.ndarray) -> np.ndarray:
    """Phase angle(D(k, t)) in radians, range (-pi, pi]. NOT unwrapped."""
    return np.angle(D)


def real_part(D: np.ndarray) -> np.ndarray:
    """Real component of D(k, t)."""
    return np.real(D)


def imag_part(D: np.ndarray) -> np.ndarray:
    """Imaginary component of D(k, t)."""
    return np.imag(D)


def reconstruct_complex(mag: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Rebuild a complex spectrum from magnitude and phase.

    ``D_reconstructed = magnitude * exp(1j * phase)``

    This should approximately equal the original ``D`` (up to
    floating point rounding) when ``mag = magnitude(D)`` and
    ``ph = phase(D)`` -- see ``tests/test_spectrum.py``.
    """
    mag = np.asarray(mag)
    ph = np.asarray(ph)
    return mag * np.exp(1j * ph)


def frequency_bins(sample_rate: float, n_fft: int) -> np.ndarray:
    """Frequency (Hz) represented by each FFT bin, standard ordering.

    ``frequency[k] = k * sample_rate / n_fft`` for ``k = 0 .. n_fft-1``.

    This follows NumPy's standard (non-fftshifted) full-complex FFT bin
    ordering: bins ``0 .. n_fft//2`` are at or below the Nyquist
    frequency (non-negative frequencies), and bins above ``n_fft//2``
    numerically correspond to negative frequencies
    (``freq - sample_rate``) even though the formula above returns a
    value greater than Nyquist for them -- this module does NOT
    rearrange bins (e.g. via ``fftshift``); callers who want an
    fftshifted / signed-frequency axis must do that themselves.

    Returns
    -------
    np.ndarray
        Shape ``(n_fft,)``, float64, in Hz, using the "aliased"
        (unshifted) convention described above.
    """
    return np.arange(n_fft, dtype=np.float64) * (sample_rate / n_fft)


def nyquist_bin_index(n_fft: int) -> int:
    """Index of the Nyquist-frequency bin (n_fft // 2)."""
    return n_fft // 2
