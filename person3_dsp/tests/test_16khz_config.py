"""Verify the Phase-1 16 kHz configuration derives the required values."""

from person3_dsp import DSPConfig


def test_16khz_derived_values():
    config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    assert config.sample_rate == 16000
    assert config.window_length == 320
    assert config.hop_length == 160
    assert config.n_fft == 320


def test_16khz_frequency_resolution():
    config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    assert config.frequency_resolution_hz() == 50.0


def test_16khz_overlap_is_50_percent():
    config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    assert config.overlap_fraction == 0.5


def test_16khz_defaults_are_explicit():
    config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    assert config.center is False
    assert config.tail_policy in ("drop", "zero_pad")
    assert config.window_type == "hann"
