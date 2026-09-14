"""Boundary condition tests: short signals, exact multiples, silence,
impulse, sine wave, and several-second signals."""

import numpy as np
import pytest

from person3_dsp import DSPConfig, STFTProcessor
from person3_dsp.framing import compute_num_frames

SAMPLE_RATE = 16000

# See test_stft_istft_identity.py: with a periodic Hann window
# (w[0] == 0) and non-centered framing, the very first sample(s) of a
# stream have no previous frame to supply overlap, so exact
# reconstruction is not guaranteed there. Documented, expected, and
# excluded from strict accuracy assertions below.
LEADING_EDGE_GUARD = 3


def _config(tail_policy="zero_pad"):
    return DSPConfig(sample_rate=SAMPLE_RATE, window_ms=20, hop_ms=10, tail_policy=tail_policy)


def test_signal_shorter_than_one_frame_zero_pad():
    config = _config("zero_pad")
    dsp = STFTProcessor(config)
    signal = np.ones(100, dtype=np.float32)  # < window_length (320)
    D = dsp.transform(signal)
    assert D.shape == (1, config.n_fft)
    reconstructed = dsp.inverse(D, output_length=len(signal))
    assert reconstructed.shape == signal.shape
    assert np.all(np.isfinite(reconstructed))


def test_signal_shorter_than_one_frame_drop():
    config = _config("drop")
    dsp = STFTProcessor(config)
    signal = np.ones(100, dtype=np.float32)
    D = dsp.transform(signal)
    assert D.shape == (0, config.n_fft)


def test_exactly_one_frame():
    # tail_policy="drop": a signal of exactly window_length samples
    # produces exactly one full frame. (With tail_policy="zero_pad",
    # the hop-advanced next frame position still falls inside the
    # signal at 50% overlap, so a second, zero-padded tail frame would
    # also be produced -- see test_non_integer_number_of_frames_* and
    # the README's overlap-add / tail-policy documentation.)
    config = _config("drop")
    dsp = STFTProcessor(config)
    signal = np.random.default_rng(0).standard_normal(config.window_length).astype(np.float32)
    D = dsp.transform(signal)
    assert D.shape == (1, config.n_fft)
    reconstructed = dsp.inverse(D, output_length=len(signal))
    assert np.all(np.isfinite(reconstructed))

    # A single, un-overlapped frame is a stricter edge case than the
    # normal streaming/multi-frame scenario: with no neighboring frame
    # to contribute energy where the Hann window tapers toward zero,
    # BOTH ends of this lone frame divide by a very small window value
    # and amplify float32 rounding error. This is expected and
    # documented (see README "Known limitations" -- accurate
    # reconstruction of an isolated single frame near its tapered
    # edges requires overlap from a neighboring frame, which by
    # definition does not exist here). The body of the frame still
    # reconstructs to numerical precision.
    body = slice(10, config.window_length - 10)
    mse = np.mean((signal[body] - reconstructed[body]) ** 2)
    assert mse < 1e-6


def test_exactly_two_frames():
    # tail_policy="drop" for the same reason as test_exactly_one_frame.
    config = _config("drop")
    dsp = STFTProcessor(config)
    n_samples = config.hop_length + config.window_length  # exactly 2 full frames, no remainder
    signal = np.random.default_rng(1).standard_normal(n_samples).astype(np.float32)
    D = dsp.transform(signal)
    assert D.shape == (2, config.n_fft)
    assert compute_num_frames(n_samples, config.window_length, config.hop_length, "drop") == 2
    # Under zero_pad, the hop-advanced next frame position (2*hop_length)
    # still falls strictly inside this signal at 50% overlap, so one
    # more (zero-padded) tail frame is produced.
    assert compute_num_frames(n_samples, config.window_length, config.hop_length, "zero_pad") == 3


def test_non_integer_number_of_frames_drop_vs_zero_pad():
    config_drop = _config("drop")
    config_pad = _config("zero_pad")
    # window=320, hop=160 -> 2 full frames cover 480 samples; add 50
    # extra samples so there's a leftover partial frame.
    n_samples = config_drop.hop_length + config_drop.window_length + 50
    signal = np.random.default_rng(2).standard_normal(n_samples).astype(np.float32)

    dsp_drop = STFTProcessor(config_drop)
    dsp_pad = STFTProcessor(config_pad)

    D_drop = dsp_drop.transform(signal)
    D_pad = dsp_pad.transform(signal)

    assert D_pad.shape[0] == D_drop.shape[0] + 1


def test_silence():
    config = _config()
    dsp = STFTProcessor(config)
    signal = np.zeros(SAMPLE_RATE, dtype=np.float32)
    D = dsp.transform(signal)
    assert np.allclose(D, 0.0)
    reconstructed = dsp.inverse(D, output_length=len(signal))
    assert np.allclose(reconstructed, 0.0, atol=1e-6)


def test_impulse():
    config = _config()
    dsp = STFTProcessor(config)
    signal = np.zeros(SAMPLE_RATE, dtype=np.float32)
    signal[1000] = 1.0
    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))
    # Energy should reappear near the impulse location.
    peak_index = int(np.argmax(np.abs(reconstructed)))
    assert abs(peak_index - 1000) <= 1
    mse = np.mean((signal - reconstructed) ** 2)
    assert mse < 1e-6


def test_sine_wave():
    config = _config()
    dsp = STFTProcessor(config)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    signal = (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))
    mse = np.mean((signal[LEADING_EDGE_GUARD:] - reconstructed[LEADING_EDGE_GUARD:]) ** 2)
    assert mse < 1e-6


def test_several_second_signal():
    config = _config()
    dsp = STFTProcessor(config)
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(SAMPLE_RATE * 4).astype(np.float32) * 0.2
    D = dsp.transform(signal)
    expected_frames = compute_num_frames(
        len(signal), config.window_length, config.hop_length, config.tail_policy
    )
    assert D.shape[0] == expected_frames
    reconstructed = dsp.inverse(D, output_length=len(signal))
    mse = np.mean((signal[LEADING_EDGE_GUARD:] - reconstructed[LEADING_EDGE_GUARD:]) ** 2)
    assert mse < 1e-6


def test_empty_signal():
    config = _config()
    dsp = STFTProcessor(config)
    signal = np.zeros(0, dtype=np.float32)
    D = dsp.transform(signal)
    assert D.shape == (0, config.n_fft)


def test_leading_edge_boundary_is_documented_and_bounded():
    """The periodic Hann window has w[0] == 0 exactly, so sample 0 of
    any stream is covered only by frame 0's zero-valued window tap and
    has no earlier frame to supply overlap. Verify this is handled
    safely (finite, bounded) rather than blowing up or producing NaN,
    and that it does not propagate beyond the documented guard."""
    config = _config()
    dsp = STFTProcessor(config)
    rng = np.random.default_rng(9)
    signal = rng.standard_normal(SAMPLE_RATE).astype(np.float32)

    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))

    assert np.all(np.isfinite(reconstructed))
    err = np.abs(signal.astype(np.float64) - reconstructed.astype(np.float64))
    # Beyond the documented guard, error should already be tiny
    # (float32/FFT-level numerical error only).
    assert np.max(err[LEADING_EDGE_GUARD:]) < 1e-4


@pytest.mark.parametrize("tail_policy", ["drop", "zero_pad"])
def test_compute_num_frames_matches_transform_shape(tail_policy):
    config = _config(tail_policy)
    dsp = STFTProcessor(config)
    for n_samples in [0, 50, 320, 321, 480, 481, 639, 640, 641, 16000]:
        signal = np.zeros(n_samples, dtype=np.float32)
        D = dsp.transform(signal)
        expected = compute_num_frames(n_samples, config.window_length, config.hop_length, tail_policy)
        assert D.shape[0] == expected, f"n_samples={n_samples}, tail_policy={tail_policy}"
