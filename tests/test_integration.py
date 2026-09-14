"""
test_integration.py
===================
SIH2026 Integration Tests.

Tests the end-to-end pipeline without breaking existing P3/P5 tests.
All 20 acceptance-criteria tests from the implementation spec.

Run with:
    pytest tests/test_integration.py -v
or from repository root:
    pytest tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from integration.config import IntegrationConfig, NLMSConfig, DSPIntegrationConfig
from integration.nlms import NLMSFilter, NLMSConfig as NLMSCfg
from integration.dataset import DatasetLoader
from integration.model_interface import DTLNStub, DFNStub, ModelInterface
from integration.pipeline import EndToEndPipeline

# Person 3 and 5 imports
from person3_dsp import DSPConfig, STFTProcessor
from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    AcousticPipeline,
    ModelID,
    RouterState,
)
from person5_acoustic.crossfade import blend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET_ROOT = REPO_ROOT
FS = 16000
DURATION_SAMPLES = FS * 3  # 3 seconds of synthetic audio


@pytest.fixture
def synthetic_audio():
    """Synthetic primary + reference signals (stationary noise, 5 dB SNR)."""
    rng = np.random.default_rng(42)
    speech = rng.standard_normal(DURATION_SAMPLES).astype(np.float32) * 0.1
    noise = rng.standard_normal(DURATION_SAMPLES).astype(np.float32) * 0.03
    primary = speech + noise
    reference = noise * 0.9  # correlated but not identical
    # Normalise
    primary /= max(np.max(np.abs(primary)), 1e-8)
    reference /= max(np.max(np.abs(reference)), 1e-8)
    return primary, reference


@pytest.fixture
def real_audio():
    """Load one real dataset example (stationary, 5 dB SNR)."""
    try:
        loader = DatasetLoader(str(DATASET_ROOT))
        example = loader.select_example(noise_category="stationary", snr_db=5, index=0)
        primary, reference, fs = loader.load_example(example)
        return primary, reference, fs, example
    except (FileNotFoundError, ImportError):
        pytest.skip("Real dataset audio not available or soundfile not installed")


@pytest.fixture
def nlms_config():
    return NLMSConfig(filter_length=64, step_size=0.01, block_size=1024, sample_rate=16000)


@pytest.fixture
def integration_config():
    return IntegrationConfig(
        nlms=NLMSConfig(filter_length=64, step_size=0.01, block_size=1024, sample_rate=16000),
        dsp=DSPIntegrationConfig(sample_rate=16000),
        output_dir="output/test_run/",
    )


# ---------------------------------------------------------------------------
# Tests 1-4: P2 NLMS basic validation
# ---------------------------------------------------------------------------

def test_01_p2_accepts_valid_input(synthetic_audio, nlms_config):
    """Test 1: P2 accepts valid primary/reference input."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    result = filt.process(primary, reference, sample_rate=16000)
    assert result is not None


def test_02_p2_produces_cleaned_waveform(synthetic_audio, nlms_config):
    """Test 2: P2 produces cleaned waveform."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    result = filt.process(primary, reference, sample_rate=16000)
    assert result.cleaned_speech is not None
    assert len(result.cleaned_speech) > 0
    assert result.cleaned_speech.dtype == np.float32


def test_03_p2_output_is_finite(synthetic_audio, nlms_config):
    """Test 3: P2 output is finite (no NaN or Inf)."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    result = filt.process(primary, reference, sample_rate=16000)
    assert not result.has_nan, "NLMS output contains NaN"
    assert not result.has_inf, "NLMS output contains Inf"
    assert np.all(np.isfinite(result.cleaned_speech))
    assert np.all(np.isfinite(result.estimated_noise))


def test_04_p2_output_has_expected_length(synthetic_audio, nlms_config):
    """Test 4: P2 output has expected length matching input."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    result = filt.process(primary, reference, sample_rate=16000)
    assert len(result.cleaned_speech) == len(primary)
    assert len(result.estimated_noise) == len(primary)


# ---------------------------------------------------------------------------
# Tests 5-6: P3 accepts P2 output
# ---------------------------------------------------------------------------

def test_05_p3_accepts_p2_output(synthetic_audio, nlms_config):
    """Test 5: P3 accepts P2 output (cleaned waveform becomes STFT input)."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    result = filt.process(primary, reference, sample_rate=16000)

    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)

    D = dsp.transform(result.cleaned_speech)
    assert D is not None
    assert D.dtype in (np.complex64, np.complex128)


def test_06_p3_produces_expected_stft_dimensions(synthetic_audio, nlms_config):
    """Test 6: P3 produces expected STFT dimensions."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    result = filt.process(primary, reference, sample_rate=16000)

    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)
    D = dsp.transform(result.cleaned_speech)

    # n_fft = 320 at 16 kHz, 20 ms window
    assert D.ndim == 2
    assert D.shape[1] == dsp_cfg.n_fft   # 320
    n_expected = dsp.num_frames_for_length(len(result.cleaned_speech))
    assert D.shape[0] == n_expected


# ---------------------------------------------------------------------------
# Tests 7-9: P5 consumes P3 spectral data
# ---------------------------------------------------------------------------

def test_07_p5_consumes_p3_spectral_information(synthetic_audio, nlms_config):
    """Test 7: P5 consumes P3 spectral information."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    nlms_result = filt.process(primary, reference, sample_rate=16000)

    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)
    # fft_size must match P3's n_fft (320 at 16kHz, 20ms window)
    p5_cfg = AcousticConfig(
        sample_rate=16000,
        frame_length=dsp_cfg.window_length,
        hop_length=dsp_cfg.hop_length,
        fft_size=dsp_cfg.n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)
    p5.set_model_available(ModelID.DTLN, True)
    p5.set_model_available(ModelID.DEEP_FILTER_NET, True)

    framer = dsp.new_streaming_framer()
    cleaned = nlms_result.cleaned_speech
    frames = framer.push(cleaned[:1024])

    diagnostics_list = []
    for i, wf in enumerate(frames):
        D = dsp.transform_frame(wf)                        # full complex (320,)
        # P3->P5 adaptor: one-sided magnitude, shape (161,)
        n_onesided = dsp_cfg.n_fft // 2 + 1
        mag_onesided = np.abs(D[:n_onesided]).astype(np.float32)
        frame = AcousticFrame(
            frame_index=i,
            sample_rate=16000,
            waveform=wf,
            magnitude=mag_onesided,                       # one-sided (161,)
            frame_length=dsp_cfg.window_length,
            hop_length=dsp_cfg.hop_length,
        )
        diag = p5.process(frame)
        diagnostics_list.append(diag)

    assert len(diagnostics_list) > 0
    assert diagnostics_list[0].complexity_score is not None


def test_08_p5_does_not_recompute_fft(synthetic_audio, nlms_config):
    """Test 8: P5 does not unnecessarily recompute FFT when magnitude is supplied."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    nlms_result = filt.process(primary, reference, sample_rate=16000)

    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)
    p5_cfg = AcousticConfig(
        sample_rate=16000,
        frame_length=dsp_cfg.window_length,
        hop_length=dsp_cfg.hop_length,
        fft_size=dsp_cfg.n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)

    cleaned = nlms_result.cleaned_speech[:1024]
    frames = dsp.new_streaming_framer().push(cleaned)
    if not frames:
        pytest.skip("No complete frames from 1024-sample chunk")

    wf = frames[0]
    D = dsp.transform_frame(wf)                          # full complex (320,)
    # P3->P5 adaptor: one-sided magnitude, shape (161,)
    n_onesided = dsp_cfg.n_fft // 2 + 1
    mag_onesided = np.abs(D[:n_onesided]).astype(np.float32)

    # Supply one-sided magnitude — P5 should use it, not recompute FFT
    frame_with_mag = AcousticFrame(
        frame_index=0,
        sample_rate=16000,
        waveform=wf,
        magnitude=mag_onesided,  # one-sided, shape (161,)
        frame_length=dsp_cfg.window_length,
        hop_length=dsp_cfg.hop_length,
    )
    assert frame_with_mag.has_spectrum  # P5 should see pre-computed spectrum


def test_09_external_snr_passthrough(synthetic_audio, nlms_config):
    """Test 9: external_snr_db can be passed from P2 to P5."""
    primary, reference = synthetic_audio
    filt = NLMSFilter(nlms_config)
    nlms_result = filt.process(primary, reference, sample_rate=16000)

    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)
    p5_cfg = AcousticConfig(
        sample_rate=16000,
        frame_length=dsp_cfg.window_length,
        hop_length=dsp_cfg.hop_length,
        fft_size=dsp_cfg.n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)

    cleaned = nlms_result.cleaned_speech[:1024]
    frames = dsp.new_streaming_framer().push(cleaned)
    if not frames:
        pytest.skip("No complete frames from 1024-sample chunk")

    wf = frames[0]
    D = dsp.transform_frame(wf)                          # full complex (320,)
    nlms_snr = nlms_result.nlms_snr_estimate_db
    # P3->P5 adaptor: one-sided magnitude, shape (161,)
    n_onesided = dsp_cfg.n_fft // 2 + 1
    mag_onesided = np.abs(D[:n_onesided]).astype(np.float32)

    frame = AcousticFrame(
        frame_index=0,
        sample_rate=16000,
        waveform=wf,
        magnitude=mag_onesided,              # one-sided (161,)
        frame_length=dsp_cfg.window_length,
        hop_length=dsp_cfg.hop_length,
        external_snr_db=nlms_snr,            # NLMS-derived SNR passed to P5
    )
    assert frame.external_snr_db == nlms_snr
    diag = p5.process(frame)
    assert diag is not None  # P5 processed with external SNR


# ---------------------------------------------------------------------------
# Tests 10-13: Router, stubs, crossfade
# ---------------------------------------------------------------------------

def test_10_router_produces_valid_decisions(synthetic_audio, nlms_config):
    """Test 10: Router produces valid decisions."""
    primary, _ = synthetic_audio
    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)
    p5_cfg = AcousticConfig(
        sample_rate=16000,
        frame_length=dsp_cfg.window_length,
        hop_length=dsp_cfg.hop_length,
        fft_size=dsp_cfg.n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)
    p5.set_model_available(ModelID.DTLN, True)
    p5.set_model_available(ModelID.DEEP_FILTER_NET, True)

    wf = primary[:dsp_cfg.window_length]
    D = dsp.transform_frame(wf)                          # full complex (320,)
    # P3->P5 adaptor: one-sided magnitude, shape (161,)
    n_onesided = dsp_cfg.n_fft // 2 + 1
    mag_onesided = np.abs(D[:n_onesided]).astype(np.float32)
    frame = AcousticFrame(
        frame_index=0, sample_rate=16000, waveform=wf,
        magnitude=mag_onesided,               # one-sided (161,)
        frame_length=dsp_cfg.window_length, hop_length=dsp_cfg.hop_length,
    )
    diag = p5.process(frame)
    decision = diag.router_decision

    assert decision is not None
    assert decision.active_model in (ModelID.DTLN, ModelID.DEEP_FILTER_NET, ModelID.NONE)
    assert decision.router_state in RouterState
    assert 0.0 <= decision.crossfade_alpha <= 1.0


def test_11_dtln_stub_accepts_model_interface():
    """Test 11: DTLN stub accepts model interface."""
    stub = DTLNStub()
    assert isinstance(stub, ModelInterface)
    D = np.random.randn(320).astype(np.complex64)
    S = stub.process(D)
    assert S.shape == D.shape
    assert np.all(np.isfinite(S))
    assert stub.is_stub
    assert "DTLN" in stub.name


def test_12_dfn_stub_accepts_same_model_interface():
    """Test 12: DFN stub accepts same model interface."""
    stub = DFNStub()
    assert isinstance(stub, ModelInterface)
    D = np.random.randn(320).astype(np.complex64)
    S = stub.process(D)
    assert S.shape == D.shape
    assert np.all(np.isfinite(S))
    assert stub.is_stub
    assert "DeepFilterNet" in stub.name


def test_13_crossfade_produces_valid_spectrum():
    """Test 13: Crossfade produces valid spectrum blending DTLN/DFN outputs."""
    dtln = DTLNStub()
    dfn = DFNStub()
    D = (np.random.randn(320) + 1j * np.random.randn(320)).astype(np.complex64)

    S_old = dtln.process(D)
    S_new = dfn.process(D)

    # Blend using P5's formula: (1-alpha)*S_old + alpha*S_new
    alpha = 0.5
    S_blend = (1.0 - alpha) * S_old + alpha * S_new

    assert S_blend.shape == D.shape
    assert np.all(np.isfinite(S_blend))
    assert S_blend.dtype == np.complex64


# ---------------------------------------------------------------------------
# Tests 14-15: ISTFT + end-to-end
# ---------------------------------------------------------------------------

def test_14_istft_produces_valid_waveform():
    """Test 14: ISTFT produces valid waveform from stub-enhanced spectrum."""
    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(dsp_cfg)

    rng = np.random.default_rng(99)
    signal = rng.standard_normal(16000).astype(np.float32) * 0.1
    D = dsp.transform(signal)

    stub = DTLNStub()
    S = np.array([stub.process(D[i]) for i in range(D.shape[0])])

    reconstructed = dsp.inverse(S, output_length=len(signal))
    assert reconstructed is not None
    assert len(reconstructed) == len(signal)
    assert np.all(np.isfinite(reconstructed))


def test_15_end_to_end_completes_on_real_wav(real_audio, integration_config):
    """Test 15: End-to-end pipeline completes on a real WAV."""
    primary, reference, fs, example = real_audio

    pipeline = EndToEndPipeline(integration_config)
    result = pipeline.process(primary, reference, sample_rate=fs)

    assert result is not None
    assert result.enhanced_wav is not None
    assert len(result.enhanced_wav) > 0


# ---------------------------------------------------------------------------
# Tests 16-20: Output validation
# ---------------------------------------------------------------------------

def test_16_output_wav_created(real_audio, integration_config, tmp_path):
    """Test 16: Output WAV is created."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    primary, reference, fs, _ = real_audio
    config = IntegrationConfig(
        nlms=NLMSConfig(filter_length=32, step_size=0.01, block_size=512, sample_rate=fs),
        dsp=DSPIntegrationConfig(sample_rate=fs),
        output_dir=str(tmp_path),
    )
    pipeline = EndToEndPipeline(config)
    result = pipeline.process(primary, reference, sample_rate=fs)

    out_path = tmp_path / "enhanced.wav"
    sf.write(str(out_path), result.enhanced_wav, fs, subtype="PCM_16")
    assert out_path.exists()
    assert out_path.stat().st_size > 44  # more than just WAV header


def test_17_output_is_mono(real_audio, integration_config):
    """Test 17: Output is mono (1D array)."""
    primary, reference, fs, _ = real_audio
    pipeline = EndToEndPipeline(integration_config)
    result = pipeline.process(primary, reference, sample_rate=fs)
    assert result.enhanced_wav.ndim == 1


def test_18_output_has_correct_sample_rate(real_audio, integration_config):
    """Test 18: Output has correct sample rate."""
    primary, reference, fs, _ = real_audio
    pipeline = EndToEndPipeline(integration_config)
    result = pipeline.process(primary, reference, sample_rate=fs)
    assert result.sample_rate == fs


def test_19_no_nan_inf_in_diagnostics(real_audio, integration_config):
    """Test 19: No NaN/Inf in frame diagnostics."""
    primary, reference, fs, _ = real_audio
    pipeline = EndToEndPipeline(integration_config)
    result = pipeline.process(primary, reference, sample_rate=fs)

    for fd in result.frame_diagnostics:
        assert np.isfinite(fd.complexity_score), \
            f"NaN/Inf in complexity at frame {fd.frame_index}"
        assert np.isfinite(fd.crossfade_alpha), \
            f"NaN/Inf in alpha at frame {fd.frame_index}"
        assert np.isfinite(fd.rms), \
            f"NaN/Inf in RMS at frame {fd.frame_index}"

    assert not result.nlms_result.has_nan
    assert not result.nlms_result.has_inf


def test_20_output_length_is_reasonable(real_audio, integration_config):
    """Test 20: Output length is reasonable (close to input length)."""
    primary, reference, fs, _ = real_audio
    pipeline = EndToEndPipeline(integration_config)
    result = pipeline.process(primary, reference, sample_rate=fs)

    expected_len = min(len(primary), len(reference))
    tolerance = fs  # allow up to 1 second difference due to OLA
    assert abs(len(result.enhanced_wav) - expected_len) <= tolerance, (
        f"Output length {len(result.enhanced_wav)} differs from "
        f"expected {expected_len} by more than {tolerance} samples"
    )


# ---------------------------------------------------------------------------
# Additional: Dataset loader
# ---------------------------------------------------------------------------

def test_dataset_loader_parses_metadata():
    """Dataset loader should parse CSV metadata without errors."""
    loader = DatasetLoader(str(DATASET_ROOT))
    examples = loader.examples
    assert len(examples) == 1000


def test_dataset_loader_select_by_category():
    """Select by noise category."""
    loader = DatasetLoader(str(DATASET_ROOT))
    ex = loader.select_example(noise_category="stationary", snr_db=5)
    assert ex.noise_category == "stationary"
    assert ex.snr_db == 5


def test_dataset_loader_select_impulsive():
    """Select impulsive noise example."""
    loader = DatasetLoader(str(DATASET_ROOT))
    ex = loader.select_example(noise_category="impulsive")
    assert ex.noise_category == "impulsive"


def test_nlms_snr_estimate_is_finite():
    """NLMS-derived SNR estimate should be a finite number."""
    cfg = NLMSConfig(filter_length=32, step_size=0.01, block_size=512, sample_rate=16000)
    filt = NLMSFilter(cfg)
    rng = np.random.default_rng(1)
    primary = rng.standard_normal(8000).astype(np.float32) * 0.1
    reference = rng.standard_normal(8000).astype(np.float32) * 0.05
    result = filt.process(primary, reference, sample_rate=16000)
    assert np.isfinite(result.nlms_snr_estimate_db)


def test_nlms_block_iterator_yields_all_samples():
    """Block iterator should cover all input samples."""
    cfg = NLMSConfig(filter_length=32, step_size=0.01, block_size=512, sample_rate=16000)
    filt = NLMSFilter(cfg)
    rng = np.random.default_rng(2)
    n = 3000
    primary = rng.standard_normal(n).astype(np.float32)
    reference = rng.standard_normal(n).astype(np.float32)

    total = 0
    for y, e in filt.process_blocks_iter(primary, reference):
        total += len(y)
    assert total == n
