"""Perfect-reconstruction test: signal -> STFT -> ISTFT -> reconstructed.

Covers random noise, a pure sine wave, a multi-frame speech-like
signal, and silence. Computes MSE, max absolute error, and
reconstruction SNR, and checks for amplitude drift / discontinuities
by inspecting sample-to-sample differences, not just aggregate error.
"""

import numpy as np
import pytest

from person3_dsp import DSPConfig, STFTProcessor


def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    noise = original - reconstructed
    signal_power = np.mean(original.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    if noise_power < 1e-20:
        return float("inf")
    if signal_power < 1e-20:
        return float("inf") if noise_power < 1e-20 else -float("inf")
    return 10.0 * np.log10(signal_power / (noise_power + 1e-20))


def _make_speech_like(rng, n_samples, sample_rate):
    # Sum of a few tones with a slowly-varying envelope, plus a little
    # noise -- not real speech, but exercises multiple frequencies and
    # multiple frames the way a speech-like signal would.
    t = np.arange(n_samples) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    tones = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 1000 * t)
    )
    noise = 0.01 * rng.standard_normal(n_samples)
    return (envelope * tones + noise).astype(np.float32)


SIGNAL_LENGTH = 16000 * 3  # 3 seconds at 16kHz
SAMPLE_RATE = 16000

# Known, documented boundary limitation (see README "Known limitations"):
# with a periodic Hann window (w[0] == 0 exactly) and non-centered,
# streaming-style framing, the very first sample(s) of a stream are
# covered by only frame 0's window, whose value at n=0 is exactly 0.
# There is no previous frame to supply overlap there, so those first
# few samples reconstruct with reduced accuracy. This is inherent to
# non-centered streaming STFT (no look-ahead padding is used) and is
# NOT a bug; the perfect-reconstruction claim applies to the signal
# body, excluding this small, documented leading edge.
LEADING_EDGE_GUARD = 3


@pytest.fixture(params=["random", "sine", "speech_like", "silence"])
def signal_case(request):
    rng = np.random.default_rng(123)
    kind = request.param
    if kind == "random":
        sig = rng.standard_normal(SIGNAL_LENGTH).astype(np.float32) * 0.1
    elif kind == "sine":
        t = np.arange(SIGNAL_LENGTH) / SAMPLE_RATE
        sig = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    elif kind == "speech_like":
        sig = _make_speech_like(rng, SIGNAL_LENGTH, SAMPLE_RATE)
    elif kind == "silence":
        sig = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    else:  # pragma: no cover
        raise ValueError(kind)
    return kind, sig


def test_stft_istft_perfect_reconstruction(signal_case):
    kind, signal = signal_case
    config = DSPConfig(sample_rate=SAMPLE_RATE, window_ms=20, hop_ms=10, tail_policy="zero_pad")
    dsp = STFTProcessor(config)

    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))

    assert reconstructed.shape == signal.shape
    assert np.all(np.isfinite(reconstructed))

    # Exclude the documented leading-edge guard (see LEADING_EDGE_GUARD
    # above) from the strict accuracy checks; still assert it is finite
    # and bounded below separately.
    body_signal = signal[LEADING_EDGE_GUARD:]
    body_reconstructed = reconstructed[LEADING_EDGE_GUARD:]

    mse = float(np.mean((body_signal.astype(np.float64) - body_reconstructed.astype(np.float64)) ** 2))
    max_abs_err = float(np.max(np.abs(body_signal.astype(np.float64) - body_reconstructed.astype(np.float64))))
    snr = _snr_db(body_signal, body_reconstructed)

    if kind == "silence":
        assert mse < 1e-10
        assert max_abs_err < 1e-4
    else:
        assert mse < 1e-6, f"{kind}: MSE too high: {mse}"
        assert max_abs_err < 5e-3, f"{kind}: max abs error too high: {max_abs_err}"
        assert snr > 60.0, f"{kind}: reconstruction SNR too low: {snr} dB"

    # Leading-edge guard region: must stay finite and bounded by the
    # signal's own amplitude range (no blow-up / NaN / inf), even
    # though exact reconstruction is not guaranteed there.
    edge_err = np.abs(signal[:LEADING_EDGE_GUARD].astype(np.float64) - reconstructed[:LEADING_EDGE_GUARD].astype(np.float64))
    assert np.all(np.isfinite(edge_err))
    assert np.all(edge_err <= 1.0 + np.max(np.abs(signal)))


def test_no_discontinuities_at_hop_boundaries(signal_case):
    """Detect clicks/discontinuities: sample-to-sample derivative of the
    reconstruction error should not spike at frame-hop boundaries."""
    kind, signal = signal_case
    if kind == "silence":
        pytest.skip("silence has no dynamics to click")
    config = DSPConfig(sample_rate=SAMPLE_RATE, window_ms=20, hop_ms=10, tail_policy="zero_pad")
    dsp = STFTProcessor(config)

    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))

    error = signal.astype(np.float64) - reconstructed.astype(np.float64)
    # Ignore the leading-edge guard and the first/last window where
    # boundary tapering is expected.
    interior = error[max(config.window_length, LEADING_EDGE_GUARD): -config.window_length]
    if interior.size < 2:
        return
    derivative = np.diff(interior)
    assert np.max(np.abs(derivative)) < 1e-2, f"{kind}: possible click/discontinuity detected"


def test_reconstruction_does_not_drift_in_amplitude():
    """Compare RMS amplitude of the first vs. last third of a long
    sine wave's reconstruction -- should be nearly identical (no slow
    amplitude drift from incorrect OLA normalization)."""
    sample_rate = 16000
    duration_s = 5
    t = np.arange(sample_rate * duration_s) / sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    config = DSPConfig(sample_rate=sample_rate, window_ms=20, hop_ms=10, tail_policy="zero_pad")
    dsp = STFTProcessor(config)
    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))

    third = len(reconstructed) // 3
    first_rms = np.sqrt(np.mean(reconstructed[:third].astype(np.float64) ** 2))
    last_rms = np.sqrt(np.mean(reconstructed[-third:].astype(np.float64) ** 2))
    assert abs(first_rms - last_rms) / first_rms < 1e-3
