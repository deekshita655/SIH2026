"""Verify D ~= |D| * exp(j*angle(D)) and real/imag/magnitude/phase consistency."""

import numpy as np
import pytest

from person3_dsp import DSPConfig, STFTProcessor
from person3_dsp import spectrum as spec


@pytest.fixture
def D():
    config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(config)
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(16000).astype(np.float32)
    return dsp.transform(signal)


def test_shape_and_dtype(D):
    assert D.ndim == 2
    assert D.shape[1] == 320
    assert D.dtype == np.complex64


def test_magnitude_phase_reconstruction(D):
    mag = spec.magnitude(D)
    ph = spec.phase(D)
    D_reconstructed = spec.reconstruct_complex(mag, ph)
    np.testing.assert_allclose(
        D_reconstructed, D, rtol=1e-4, atol=1e-4,
        err_msg="magnitude * exp(j*phase) should approximately reconstruct D",
    )


def test_real_imag_consistency(D):
    real = spec.real_part(D)
    imag = spec.imag_part(D)
    reconstructed = real + 1j * imag
    np.testing.assert_allclose(reconstructed, D, rtol=1e-6, atol=1e-6)


def test_magnitude_matches_sqrt_real2_imag2(D):
    mag = spec.magnitude(D)
    expected = np.sqrt(spec.real_part(D) ** 2 + spec.imag_part(D) ** 2)
    np.testing.assert_allclose(mag, expected, rtol=1e-4, atol=1e-5)


def test_phase_range(D):
    ph = spec.phase(D)
    assert np.all(ph > -np.pi - 1e-6)
    assert np.all(ph <= np.pi + 1e-6)


def test_frequency_bins_convention():
    freqs = spec.frequency_bins(16000, 320)
    assert freqs.shape == (320,)
    assert freqs[0] == 0.0
    assert freqs[1] == 50.0
    assert freqs[160] == 8000.0  # Nyquist bin


def test_full_complex_spectrum_not_onesided(D):
    # A one-sided (rFFT-style) spectrum would have n_fft // 2 + 1 bins.
    # We require the FULL complex spectrum: all n_fft bins present.
    assert D.shape[1] == 320
    assert D.shape[1] != 320 // 2 + 1
