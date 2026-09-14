"""Verify the same code derives the correct 48 kHz production configuration.

Critically: 320/160 must NOT be hard-coded anywhere -- these values
must come purely from sample_rate * window_ms / hop_ms.
"""

from person3_dsp import DSPConfig, STFTProcessor
import numpy as np


def test_48khz_derived_values():
    config = DSPConfig(sample_rate=48000, window_ms=20, hop_ms=10)
    assert config.sample_rate == 48000
    assert config.window_length == 960
    assert config.hop_length == 480
    assert config.n_fft == 960


def test_48khz_frequency_resolution():
    config = DSPConfig(sample_rate=48000, window_ms=20, hop_ms=10)
    assert abs(config.frequency_resolution_hz() - 50.0) < 1e-9


def test_48khz_stft_shape():
    config = DSPConfig(sample_rate=48000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(config)
    signal = np.random.default_rng(0).standard_normal(48000 * 2).astype(np.float32)
    D = dsp.transform(signal)
    assert D.shape[1] == 960
    assert D.dtype == np.complex64


def test_20ms_is_not_320_samples_at_48khz():
    # Regression guard against the exact mistake the spec warns about:
    # 320 samples is 20ms at 16kHz, NOT at 48kHz.
    config = DSPConfig(sample_rate=48000, window_ms=20, hop_ms=10)
    assert config.window_length != 320
    assert config.window_length == 960
