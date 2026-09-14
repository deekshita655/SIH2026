"""
windowing.py
============

Owner: Person 3.

Isolated window-generation logic, as required by the spec ("Window
generation must be isolated in windowing.py").

Only a *periodic* (a.k.a. DFT-even) window is used, i.e. the
mathematical equivalent of ``scipy.signal.windows.hann(N, sym=False)``.

Why periodic and not symmetric?
--------------------------------
A symmetric Hann window (``sym=True``, the default in many libraries
and in ``numpy.hanning``) has its first and last samples equal and
duplicates that shape when tiled -- it is designed for FIR filter
design, not for overlap-add analysis/synthesis. For STFT/OLA
processing the periodic/DFT-even variant is the mathematically
correct choice because it satisfies the constant-overlap-add (COLA)
property at standard hop sizes (e.g. 50% overlap), which is required
for correct, ripple-free reconstruction. Using the symmetric window
here would introduce a small but real amplitude modulation artifact
into any reconstructed signal. This module always uses the periodic
form and documents that choice explicitly, as required.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal.windows import hann as _scipy_hann

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only if scipy is absent
    _HAVE_SCIPY = False


_SUPPORTED_WINDOW_TYPES = {"hann"}


def _periodic_hann(n: int) -> np.ndarray:
    """Fallback periodic/DFT-even Hann window without scipy.

    Mathematically identical to ``scipy.signal.windows.hann(n, sym=False)``:
    computed as the symmetric Hann of length n+1 with the last sample
    dropped, which is exactly what ``sym=False`` does internally.
    """
    if n == 1:
        return np.ones(1, dtype=np.float64)
    k = np.arange(n)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
    return w


def get_window(window_type: str, length: int) -> np.ndarray:
    """Return a periodic (DFT-even) window of the given type and length.

    Parameters
    ----------
    window_type:
        Currently only ``"hann"`` is supported.
    length:
        Window length in samples (``window_length`` from ``DSPConfig``,
        NOT ``n_fft`` -- callers are responsible for any zero-padding
        up to ``n_fft``).

    Returns
    -------
    np.ndarray
        1D float64 array of shape ``(length,)``. Callers should cast
        to the desired working dtype (e.g. float32) as needed.
    """
    if window_type not in _SUPPORTED_WINDOW_TYPES:
        raise ValueError(
            f"Unsupported window_type {window_type!r}; "
            f"supported types: {sorted(_SUPPORTED_WINDOW_TYPES)}"
        )
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    if _HAVE_SCIPY:
        w = _scipy_hann(length, sym=False)
    else:  # pragma: no cover
        w = _periodic_hann(length)
    return np.asarray(w, dtype=np.float64)


def get_analysis_window(window_type: str, window_length: int, dtype=np.float32) -> np.ndarray:
    """Convenience wrapper: periodic window cast to the working dtype.

    This is the function most callers (framing/STFT/ISTFT) should use.
    """
    w = get_window(window_type, window_length)
    return w.astype(dtype)
