"""
test_features.py
================
Comprehensive tests for person5_acoustic.features module.
"""

import math
import pytest
import numpy as np

from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    FeatureExtractor,
)
from person5_acoustic.features import (
    _rms,
    _zcr,
    _spectral_flux,
    _spectral_entropy,
    _spectral_centroid,
    _build_freq_bins,
    TransientDetector,
    RuntimeSNREstimator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config16():
    return AcousticConfig.development_16khz()


@pytest.fixture
def config48():
    return AcousticConfig.production_48khz()


@pytest.fixture
def extractor16(config16):
    return FeatureExtractor(config16)


def _sine_frame(freq_hz: float, sr: int, n: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _white_noise(n: int, amp: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amp).astype(np.float32)


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# RMS tests
# ---------------------------------------------------------------------------

class TestRMS:
    def test_silence_is_zero(self):
        assert _rms(_silence(320)) == pytest.approx(0.0)

    def test_unit_sine(self):
        # sin wave with amplitude 1 → RMS = 1/sqrt(2)
        t = np.linspace(0, 2 * np.pi, 10000)
        x = np.sin(t).astype(np.float32)
        assert _rms(x) == pytest.approx(1.0 / math.sqrt(2), abs=1e-3)

    def test_dc_signal(self):
        x = np.ones(320, dtype=np.float32) * 0.5
        assert _rms(x) == pytest.approx(0.5)

    def test_empty_array(self):
        assert _rms(np.array([], dtype=np.float32)) == 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.standard_normal(320).astype(np.float32)
            assert _rms(x) >= 0.0

    def test_scales_with_amplitude(self):
        x = _sine_frame(440, 16000, 320, amp=1.0)
        y = _sine_frame(440, 16000, 320, amp=2.0)
        assert _rms(y) == pytest.approx(_rms(x) * 2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# ZCR tests
# ---------------------------------------------------------------------------

class TestZCR:
    def test_silence_is_zero(self):
        assert _zcr(_silence(320)) == 0.0

    def test_alternating_sign_is_one(self):
        x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        assert _zcr(x) == pytest.approx(1.0)

    def test_dc_is_zero(self):
        x = np.ones(100, dtype=np.float32) * 0.5
        assert _zcr(x) == pytest.approx(0.0)

    def test_bounded(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            x = rng.standard_normal(320).astype(np.float32)
            z = _zcr(x)
            assert 0.0 <= z <= 1.0

    def test_high_freq_signal_has_high_zcr(self):
        # Nyquist (alternating) → max ZCR
        x = np.array([(1 if i % 2 == 0 else -1) for i in range(320)], dtype=np.float32)
        assert _zcr(x) == pytest.approx(1.0)

    def test_single_sample(self):
        assert _zcr(np.array([1.0], dtype=np.float32)) == 0.0

    def test_two_samples_crossing(self):
        x = np.array([1.0, -1.0], dtype=np.float32)
        assert _zcr(x) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Spectral Flux tests
# ---------------------------------------------------------------------------

class TestSpectralFlux:
    def test_first_frame_is_zero(self):
        mag = np.ones(257, dtype=np.float32)
        assert _spectral_flux(mag, None) == pytest.approx(0.0)

    def test_identical_frames_zero_flux(self):
        mag = np.ones(257, dtype=np.float32) * 0.5
        flux = _spectral_flux(mag, mag.copy())
        assert flux == pytest.approx(0.0)

    def test_flux_nonnegative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.random(257).astype(np.float32)
            b = rng.random(257).astype(np.float32)
            assert _spectral_flux(a, b) >= 0.0

    def test_flux_increases_with_change(self):
        mag_a = np.zeros(257, dtype=np.float32)
        mag_b = np.ones(257, dtype=np.float32)
        mag_c = np.ones(257, dtype=np.float32) * 2.0
        flux_ab = _spectral_flux(mag_b, mag_a)
        flux_ac = _spectral_flux(mag_c, mag_a)
        assert flux_ac > flux_ab

    def test_symmetry(self):
        # Flux(a,b) == Flux(b,a) because we compute L2 norm of difference
        rng = np.random.default_rng(1)
        a = rng.random(100).astype(np.float32)
        b = rng.random(100).astype(np.float32)
        assert _spectral_flux(a, b) == pytest.approx(_spectral_flux(b, a))


# ---------------------------------------------------------------------------
# Spectral Entropy tests
# ---------------------------------------------------------------------------

class TestSpectralEntropy:
    def test_zero_energy_returns_zero(self):
        mag = np.zeros(257, dtype=np.float32)
        assert _spectral_entropy(mag) == pytest.approx(0.0)

    def test_flat_spectrum_is_max_entropy(self):
        # Flat magnitude spectrum → maximum normalized entropy ≈ 1.0
        k = 256
        mag = np.ones(k, dtype=np.float32)
        h = _spectral_entropy(mag)
        assert h == pytest.approx(1.0, abs=1e-5)

    def test_impulse_spectrum_is_low_entropy(self):
        # Single non-zero bin → low entropy
        k = 256
        mag = np.zeros(k, dtype=np.float32)
        mag[50] = 1.0
        h = _spectral_entropy(mag)
        assert h == pytest.approx(0.0, abs=1e-5)

    def test_bounded(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            mag = np.abs(rng.standard_normal(257)).astype(np.float32)
            h = _spectral_entropy(mag)
            assert 0.0 <= h <= 1.0 + 1e-8

    def test_normalized_by_log_k(self):
        # Verify H / log(K) formula
        k = 100
        mag = np.ones(k, dtype=np.float32)
        h = _spectral_entropy(mag)
        # For flat spectrum: H = log(K), H_norm = H / log(K) = 1
        assert h == pytest.approx(1.0, abs=1e-5)

    def test_single_bin(self):
        mag = np.array([1.0], dtype=np.float32)
        assert _spectral_entropy(mag) == 0.0


# ---------------------------------------------------------------------------
# Spectral Centroid tests
# ---------------------------------------------------------------------------

class TestSpectralCentroid:
    def test_low_freq_heavy(self):
        k = 10
        freq_bins = np.linspace(0, 8000, k)
        mag = np.zeros(k)
        mag[0] = 1.0  # all energy at DC
        sc = _spectral_centroid(mag, freq_bins)
        assert sc == pytest.approx(0.0)

    def test_high_freq_heavy(self):
        k = 10
        freq_bins = np.linspace(0, 8000, k)
        mag = np.zeros(k)
        mag[-1] = 1.0  # all energy at Nyquist
        sc = _spectral_centroid(mag, freq_bins)
        assert sc == pytest.approx(8000.0)

    def test_symmetric_flat_spectrum(self):
        k = 11
        freq_bins = np.linspace(0, 10000, k)
        mag = np.ones(k)
        sc = _spectral_centroid(mag, freq_bins)
        assert sc == pytest.approx(5000.0, abs=1.0)

    def test_zero_energy_fallback(self):
        k = 10
        freq_bins = np.linspace(0, 8000, k)
        mag = np.zeros(k)
        sc = _spectral_centroid(mag, freq_bins)
        # Should return mean of freq_bins (safe fallback)
        assert sc == pytest.approx(np.mean(freq_bins))

    def test_sample_rate_aware_16khz(self):
        config16 = AcousticConfig.development_16khz()
        freq = _build_freq_bins(config16.fft_size, config16.sample_rate)
        assert freq[-1] == pytest.approx(8000.0)

    def test_sample_rate_aware_48khz(self):
        config48 = AcousticConfig.production_48khz()
        freq = _build_freq_bins(config48.fft_size, config48.sample_rate)
        assert freq[-1] == pytest.approx(24000.0)


# ---------------------------------------------------------------------------
# Transient Detector tests
# ---------------------------------------------------------------------------

class TestTransientDetector:
    def test_no_transient_on_steady_signal(self):
        config = AcousticConfig.development_16khz()
        det = TransientDetector(config)
        for _ in range(20):
            score, flag = det.update(0.1, 0.1)
        # Steady signal should not flag transients
        assert not flag

    def test_transient_on_sudden_spike(self):
        config = AcousticConfig.development_16khz()
        det = TransientDetector(config)
        # Warm up with quiet signal
        for _ in range(10):
            det.update(0.01, 0.01)
        # Sudden loud spike
        score, flag = det.update(5.0, 5.0)
        assert flag

    def test_score_bounded(self):
        config = AcousticConfig.development_16khz()
        det = TransientDetector(config)
        det.update(0.1, 0.1)
        for _ in range(5):
            score, _ = det.update(0.1, 0.1)
            assert 0.0 <= score <= 1.0

    def test_first_frame_no_transient(self):
        config = AcousticConfig.development_16khz()
        det = TransientDetector(config)
        score, flag = det.update(0.5, 0.5)
        assert not flag
        assert score == pytest.approx(0.0)

    def test_reset_clears_baseline(self):
        config = AcousticConfig.development_16khz()
        det = TransientDetector(config)
        for _ in range(10):
            det.update(0.1, 0.1)
        det.reset()
        # After reset, first call should not flag transient
        score, flag = det.update(0.1, 0.1)
        assert not flag


# ---------------------------------------------------------------------------
# FeatureExtractor integration tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_processes_waveform_only(self, config16):
        ext = FeatureExtractor(config16)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=16000,
            waveform=_sine_frame(440, 16000, 320),
        )
        fv = ext.process(frame)
        assert fv.frame_index == 0
        assert fv.rms > 0
        assert 0.0 <= fv.zcr <= 1.0
        assert not fv.is_zero_energy

    def test_processes_spectrum_only(self, config16):
        ext = FeatureExtractor(config16)
        rng = np.random.default_rng(0)
        mag = np.abs(rng.standard_normal(257)).astype(np.float32)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=16000,
            magnitude=mag,
        )
        fv = ext.process(frame)
        assert fv.spectral_entropy >= 0.0

    def test_zero_energy_detection(self, config16):
        ext = FeatureExtractor(config16)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=16000,
            waveform=_silence(320),
        )
        fv = ext.process(frame)
        assert fv.is_zero_energy
        assert fv.rms < 1e-9

    def test_centroid_variation_zero_on_first_frame(self, config16):
        ext = FeatureExtractor(config16)
        rng = np.random.default_rng(5)
        mag = np.abs(rng.standard_normal(257)).astype(np.float32)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=16000,
            waveform=_sine_frame(1000, 16000, 320),
            magnitude=mag,
        )
        fv = ext.process(frame)
        assert fv.centroid_variation == pytest.approx(0.0)

    def test_centroid_variation_nonzero_on_second_frame(self, config16):
        ext = FeatureExtractor(config16)
        rng = np.random.default_rng(9)
        # Frame 1: low freq
        mag1 = np.zeros(257, dtype=np.float32)
        mag1[10] = 1.0
        frame1 = AcousticFrame(
            frame_index=0, sample_rate=16000, magnitude=mag1,
            waveform=_sine_frame(200, 16000, 320),
        )
        ext.process(frame1)
        # Frame 2: high freq
        mag2 = np.zeros(257, dtype=np.float32)
        mag2[200] = 1.0
        frame2 = AcousticFrame(
            frame_index=1, sample_rate=16000, magnitude=mag2,
            waveform=_sine_frame(7000, 16000, 320),
        )
        fv2 = ext.process(frame2)
        assert fv2.centroid_variation > 0.0

    def test_external_snr_passthrough(self, config16):
        ext = FeatureExtractor(config16)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=16000,
            waveform=_sine_frame(440, 16000, 320),
            external_snr_db=20.0,
        )
        fv = ext.process(frame)
        assert fv.estimated_snr_db == pytest.approx(20.0)

    def test_spectral_flux_zero_on_first_frame(self, config16):
        ext = FeatureExtractor(config16)
        mag = np.ones(257, dtype=np.float32)
        frame = AcousticFrame(
            frame_index=0, sample_rate=16000, magnitude=mag,
            waveform=_sine_frame(440, 16000, 320),
        )
        fv = ext.process(frame)
        assert fv.spectral_flux == pytest.approx(0.0)

    def test_48khz_config(self, config48):
        ext = FeatureExtractor(config48)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=48000,
            waveform=_white_noise(960).astype(np.float32),
            frame_length=960,
            hop_length=480,
        )
        fv = ext.process(frame)
        assert fv.rms > 0

    def test_reset_clears_previous_magnitude(self, config16):
        ext = FeatureExtractor(config16)
        mag1 = np.ones(257, dtype=np.float32)
        frame1 = AcousticFrame(frame_index=0, sample_rate=16000, magnitude=mag1,
                               waveform=_sine_frame(440, 16000, 320))
        ext.process(frame1)
        ext.reset()
        frame2 = AcousticFrame(frame_index=1, sample_rate=16000, magnitude=mag1,
                               waveform=_sine_frame(440, 16000, 320))
        fv2 = ext.process(frame2)
        assert fv2.spectral_flux == pytest.approx(0.0)  # after reset, no prev mag
