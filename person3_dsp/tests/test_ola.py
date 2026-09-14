"""Test overlap-add (ola.py) independently of STFT/FFT."""

import numpy as np
import pytest

from person3_dsp.ola import overlap_add
from person3_dsp.windowing import get_analysis_window


def test_ola_reconstructs_constant_signal_with_hann_50pct_overlap():
    # A DFT-even Hann window at 50% overlap is well known to satisfy
    # constant-overlap-add for the window itself; here we verify that
    # windowing a constant signal and running it through analysis
    # window * synthesis window normalization recovers the constant
    # signal (this is exactly what STFT/ISTFT relies on).
    window_length = 320
    hop_length = 160
    window = get_analysis_window("hann", window_length)

    num_frames = 20
    const_value = 3.0
    frames = np.full((num_frames, window_length), const_value, dtype=np.float32)
    windowed_frames = frames * window[np.newaxis, :]

    output, window_accum = overlap_add(
        frames_time=windowed_frames,
        hop_length=hop_length,
        synthesis_window=window,
        analysis_window=window,
    )

    # Interior region (away from the very first/last frame's edge
    # effects) should reconstruct the constant value closely.
    interior = output[window_length: -window_length]
    np.testing.assert_allclose(interior, const_value, rtol=1e-3, atol=1e-3)


def test_ola_no_division_by_zero_when_windows_are_all_zero():
    window_length = 64
    hop_length = 32
    zero_window = np.zeros(window_length, dtype=np.float32)
    frames = np.random.default_rng(1).standard_normal((3, window_length)).astype(np.float32)

    output, window_accum = overlap_add(
        frames_time=frames,
        hop_length=hop_length,
        synthesis_window=zero_window,
        analysis_window=zero_window,
        eps=1e-8,
    )
    assert np.all(np.isfinite(output))
    assert np.all(window_accum == 0.0)


def test_ola_output_length_truncation_and_padding():
    window_length = 32
    hop_length = 16
    window = get_analysis_window("hann", window_length)
    frames = np.random.default_rng(2).standard_normal((5, window_length)).astype(np.float32)

    natural_length = (5 - 1) * hop_length + window_length
    out_natural, _ = overlap_add(frames, hop_length, window, window)
    assert out_natural.shape[0] == natural_length

    out_short, _ = overlap_add(frames, hop_length, window, window, output_length=natural_length - 10)
    assert out_short.shape[0] == natural_length - 10

    out_long, _ = overlap_add(frames, hop_length, window, window, output_length=natural_length + 10)
    assert out_long.shape[0] == natural_length + 10
    # padded tail should be exactly zero
    np.testing.assert_array_equal(out_long[natural_length:], 0.0)


def test_ola_single_frame_matches_windowed_frame_over_itself():
    window_length = 16
    hop_length = 8
    window = get_analysis_window("hann", window_length)
    single_signal = np.random.default_rng(3).standard_normal(window_length).astype(np.float32)
    frame = (single_signal * window)[np.newaxis, :]

    output, window_accum = overlap_add(frame, hop_length, window, window)
    # output = (signal*window*window) / (window*window), which equals
    # signal wherever window is non-zero.
    nonzero = window_accum > 1e-6
    np.testing.assert_allclose(output[nonzero], single_signal[nonzero], rtol=1e-4, atol=1e-4)


def test_ola_shape_validation_errors():
    window = get_analysis_window("hann", 32)
    bad_frames_1d = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError):
        overlap_add(bad_frames_1d, 16, window, window)

    frames = np.zeros((2, 32), dtype=np.float32)
    wrong_window = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError):
        overlap_add(frames, 16, wrong_window, window)
