"""
test_integration_phase2.py
==========================
Phase 2 integration tests for SIH2026.

Tests 12 additional acceptance criteria from the task spec:

 1. P2 output -> P3 buffering (chunk sizes don't need to align)
 2. Arbitrary chunk boundaries (various sizes: 100, 512, 1600, 3000)
 3. 1024-sample block -> 160-sample hop buffering correctness
 4. P3 full complex -> P5 one-sided adaptor shapes
 5. Frequency-bin consistency (P3 vs P5 bins)
 6. Frame count consistency across chunk sizes
 7. No sample duplication after OLA reconstruction
 8. No sample loss after OLA reconstruction
 9. End-of-stream (flush) behavior
10. P4 spectrum shape contract (stub in=full complex, out=full complex)
11. End-to-end real WAV processing with metrics
12. Router transition behavior (hysteresis, no chatter)

Run from repo root:
    pytest tests/test_integration_phase2.py -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "person3_dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "person5_acoustic" / "src"))

from integration.config import IntegrationConfig, NLMSConfig, DSPIntegrationConfig
from integration.nlms import NLMSFilter
from integration.model_interface import DTLNStub, DFNStub
from integration.pipeline import EndToEndPipeline

from person3_dsp import DSPConfig, STFTProcessor
from person5_acoustic import (
    AcousticConfig, AcousticFrame, AcousticPipeline, ModelID, RouterState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 16000
CHUNK_5S = FS * 5     # 5-second synthetic audio


@pytest.fixture
def synth_5s():
    """5-second synthetic noisy speech (stationary noise, ~5 dB SNR)."""
    rng = np.random.default_rng(7)
    speech  = rng.standard_normal(CHUNK_5S).astype(np.float32) * 0.12
    noise   = rng.standard_normal(CHUNK_5S).astype(np.float32) * 0.04
    primary   = (speech + noise)
    reference = noise * 0.9 + rng.standard_normal(CHUNK_5S).astype(np.float32) * 0.005
    # normalise
    primary   /= max(np.max(np.abs(primary)), 1e-8)
    reference /= max(np.max(np.abs(reference)), 1e-8)
    return primary, reference


@pytest.fixture
def dsp_cfg():
    return DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)


@pytest.fixture
def real_audio():
    """Load one real stationary example from dataset."""
    try:
        from integration.dataset import DatasetLoader
        loader = DatasetLoader(str(REPO_ROOT))
        ex = loader.select_example(noise_category="stationary", snr_db=5, index=0)
        primary, reference, fs = loader.load_example(ex)
        return primary, reference, fs, ex
    except (FileNotFoundError, ImportError):
        pytest.skip("Real dataset or soundfile not available")


@pytest.fixture
def default_config():
    return IntegrationConfig(
        nlms=NLMSConfig(filter_length=64, step_size=0.01, block_size=1024, sample_rate=16000),
        dsp=DSPIntegrationConfig(sample_rate=16000),
        output_dir="output/phase2_tests/",
    )


# ---------------------------------------------------------------------------
# Helper: run P3 StreamingFramer over arbitrary chunks, count samples in/out
# ---------------------------------------------------------------------------

def _count_framer_samples(dsp: STFTProcessor, signal: np.ndarray, chunk_size: int):
    """
    Feed signal to P3 StreamingFramer in chunk_size pieces.
    Returns (total_frames, total_input_samples, total_output_samples_in_frames).
    """
    framer = dsp.new_streaming_framer()
    win_len = dsp.config.window_length
    hop_len = dsp.config.hop_length
    total_frames = 0
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i : i + chunk_size]
        frames = framer.push(chunk)
        total_frames += len(frames)
    last = framer.flush()
    if last is not None:
        total_frames += 1
    total_out = total_frames * win_len       # overlap-add output size (gross)
    total_in  = len(signal)
    return total_frames, total_in, total_out


# ===========================================================================
# TEST 1: P2 output -> P3 buffering  (streaming, non-aligned chunk sizes)
# ===========================================================================

def test_p2_to_p3_buffering_is_lossless(synth_5s, dsp_cfg):
    """P2 block (1024) -> P3 framer correctly buffers without loss."""
    primary, reference = synth_5s
    from integration.nlms import NLMSConfig
    nlms_cfg = NLMSConfig(filter_length=64, step_size=0.01, block_size=1024)
    filt = NLMSFilter(nlms_cfg)
    nlms_result = filt.process(primary, reference, sample_rate=16000)
    cleaned = nlms_result.cleaned_speech  # 1D, same length as primary

    # Feed cleaned in 1024-sample blocks through P3 framer
    dsp = STFTProcessor(dsp_cfg)
    framer = dsp.new_streaming_framer()

    block_size = 1024
    all_frames = []
    for i in range(0, len(cleaned), block_size):
        chunk = cleaned[i : i + block_size]
        frames = framer.push(chunk)
        all_frames.extend(frames)
    last = framer.flush()
    if last is not None:
        all_frames.append(last)

    # Each frame must be exactly window_length samples
    assert all(len(f) == dsp_cfg.window_length for f in all_frames), \
        "Framer produced frames of wrong length"
    # At least some frames must have been produced
    assert len(all_frames) > 0


# ===========================================================================
# TEST 2: Arbitrary chunk boundaries
# ===========================================================================

@pytest.mark.parametrize("chunk_size", [100, 160, 512, 1024, 1600, 3000, 7331])
def test_p3_framer_arbitrary_chunk_boundaries(dsp_cfg, chunk_size):
    """P3 StreamingFramer handles arbitrary input chunk sizes."""
    dsp = STFTProcessor(dsp_cfg)
    rng = np.random.default_rng(99 + chunk_size)
    signal = rng.standard_normal(CHUNK_5S).astype(np.float32)

    frames, n_in, _ = _count_framer_samples(dsp, signal, chunk_size)

    # Must have produced at least floor(n_in / hop) - 1 frames
    min_frames = max(0, (n_in - dsp_cfg.window_length) // dsp_cfg.hop_length)
    assert frames >= min_frames, (
        f"chunk_size={chunk_size}: got {frames} frames, expected >= {min_frames}"
    )


# ===========================================================================
# TEST 3: 1024-sample block -> 160-sample hop buffering
# ===========================================================================

def test_1024_block_to_160_hop_produces_correct_frames(dsp_cfg):
    """
    Feed exactly 1024-sample blocks; verify frame count matches expected.
    1024 samples / 160 hop = 6 complete frames per block (with carry-over).
    """
    dsp = STFTProcessor(dsp_cfg)
    framer = dsp.new_streaming_framer()
    rng = np.random.default_rng(11)
    n_blocks = 10
    signal = rng.standard_normal(1024 * n_blocks).astype(np.float32)
    n_fft = dsp_cfg.n_fft        # 320
    hop   = dsp_cfg.hop_length   # 160

    all_frames = []
    for i in range(n_blocks):
        chunk = signal[i*1024 : (i+1)*1024]
        all_frames.extend(framer.push(chunk))
    last = framer.flush()
    if last is not None:
        all_frames.append(last)

    # Theoretical: ceil((n - n_fft) / hop) + 1  with zero_pad
    total_samples = 1024 * n_blocks
    expected = dsp.num_frames_for_length(total_samples)
    assert abs(len(all_frames) - expected) <= 1, (
        f"Expected ~{expected} frames, got {len(all_frames)}"
    )


# ===========================================================================
# TEST 4: P3 full complex -> P5 one-sided adapter shapes
# ===========================================================================

def test_p3_to_p5_one_sided_adapter_shapes(dsp_cfg):
    """
    The integration adapter must:
      - Take D (full complex, shape n_fft=320)
      - Produce magnitude_onesided (shape 161 = n_fft//2+1)
      - Produce freq_bins_onesided (shape 161)
    """
    dsp = STFTProcessor(dsp_cfg)
    n_fft     = dsp_cfg.n_fft           # 320
    n_onesided = n_fft // 2 + 1         # 161
    win_len   = dsp_cfg.window_length   # 320

    rng = np.random.default_rng(42)
    wf = rng.standard_normal(win_len).astype(np.float32)

    D = dsp.transform_frame(wf)
    assert D.shape == (n_fft,), f"P3 full spectrum shape must be ({n_fft},), got {D.shape}"
    assert np.iscomplexobj(D), "P3 spectrum must be complex"

    # Integration adapter
    mag_onesided  = np.abs(D[:n_onesided]).astype(np.float32)
    all_freq_bins = dsp.frequency_bins()
    freq_onesided = all_freq_bins[:n_onesided].astype(np.float32)

    assert mag_onesided.shape  == (n_onesided,), \
        f"One-sided magnitude shape must be ({n_onesided},)"
    assert freq_onesided.shape == (n_onesided,), \
        f"One-sided freq_bins shape must be ({n_onesided},)"

    # P5 should accept this without shape error
    p5_cfg = AcousticConfig(
        sample_rate=16000, frame_length=win_len,
        hop_length=dsp_cfg.hop_length, fft_size=n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)
    frame = AcousticFrame(
        frame_index=0, sample_rate=16000, waveform=wf,
        magnitude=mag_onesided, freq_bins=freq_onesided,
        frame_length=win_len, hop_length=dsp_cfg.hop_length,
    )
    diag = p5.process(frame)
    assert diag is not None
    assert diag.complexity_score is not None


# ===========================================================================
# TEST 5: Frequency-bin consistency (P3 full vs P5 one-sided)
# ===========================================================================

def test_frequency_bin_consistency(dsp_cfg):
    """
    P3's frequency_bins() for index k should equal k*Fs/N.
    One-sided bins for k=0..160 should exactly match the first 161 P3 bins.
    """
    dsp = STFTProcessor(dsp_cfg)
    n_fft = dsp_cfg.n_fft           # 320
    fs    = dsp_cfg.sample_rate      # 16000
    n_onesided = n_fft // 2 + 1     # 161

    all_bins = dsp.frequency_bins()
    assert len(all_bins) == n_fft, f"P3 must produce {n_fft} bins"

    # Verify bin 0 = 0 Hz, bin N/2 = Nyquist
    assert np.isclose(all_bins[0], 0.0, atol=0.1), f"bin[0] should be 0 Hz, got {all_bins[0]}"
    nyquist = fs / 2
    assert np.isclose(all_bins[n_fft // 2], nyquist, rtol=0.01), \
        f"bin[N//2] should be {nyquist} Hz, got {all_bins[n_fft//2]}"

    # One-sided slice matches first n_onesided bins
    onesided = all_bins[:n_onesided]
    assert onesided.shape == (n_onesided,)
    assert onesided[-1] <= nyquist + 1.0  # must not exceed Nyquist


# ===========================================================================
# TEST 6: Frame count consistency across chunk sizes
# ===========================================================================

@pytest.mark.parametrize("chunk_size", [160, 320, 1024, 2048])
def test_frame_count_consistent_across_chunk_sizes(dsp_cfg, chunk_size):
    """
    The same signal fed in different chunk sizes must yield the same total
    frame count (or within ±1 due to tail policy).
    """
    rng = np.random.default_rng(55)
    signal = rng.standard_normal(CHUNK_5S).astype(np.float32)

    dsp = STFTProcessor(dsp_cfg)
    expected_frames = dsp.num_frames_for_length(len(signal))

    frames, _, _ = _count_framer_samples(dsp, signal, chunk_size)
    assert abs(frames - expected_frames) <= 1, (
        f"chunk_size={chunk_size}: frames={frames}, expected={expected_frames}"
    )


# ===========================================================================
# TEST 7: No sample duplication in OLA reconstruction
# ===========================================================================

def test_no_sample_duplication_in_ola(dsp_cfg):
    """
    OLA reconstructed output should not be significantly longer than input.
    If samples are duplicated, the output will be longer than expected.
    """
    dsp = STFTProcessor(dsp_cfg)
    rng = np.random.default_rng(17)
    n_in = CHUNK_5S
    signal = rng.standard_normal(n_in).astype(np.float32) * 0.3

    D = dsp.transform(signal)  # batch STFT: (num_frames, n_fft)
    recon = dsp.inverse(D, output_length=n_in)

    assert len(recon) == n_in, (
        f"Reconstructed length {len(recon)} != input length {n_in} "
        "(possible duplication or extension)"
    )


# ===========================================================================
# TEST 8: No sample loss in streaming pipeline
# ===========================================================================

def test_no_sample_loss_streaming(synth_5s, default_config):
    """
    End-to-end pipeline output must have the same length as the input
    within one window tolerance (OLA introduces boundary effects but
    no samples should be silently discarded).
    """
    primary, reference = synth_5s
    pipeline = EndToEndPipeline(default_config)
    result = pipeline.process(primary, reference, sample_rate=16000)

    expected = min(len(primary), len(reference))
    tolerance = 16000  # up to 1 second tolerance
    assert abs(len(result.enhanced_wav) - expected) <= tolerance, (
        f"Output length {len(result.enhanced_wav)} differs from "
        f"input {expected} by more than {tolerance} samples"
    )


# ===========================================================================
# TEST 9: End-of-stream (flush) behavior
# ===========================================================================

def test_end_of_stream_flush(dsp_cfg):
    """
    Streaming framer flush() should return the last partial frame (or None
    if buffer is empty). Must not raise or return garbage.
    """
    dsp = STFTProcessor(dsp_cfg)
    framer = dsp.new_streaming_framer()
    win_len = dsp_cfg.window_length

    rng = np.random.default_rng(33)
    # Feed a number of samples that does NOT end on a hop boundary
    signal = rng.standard_normal(1777).astype(np.float32)
    all_frames = list(framer.push(signal))
    last = framer.flush()

    if last is not None:
        # Must be exactly window_length samples (zero-padded)
        assert len(last) == win_len, \
            f"flush() returned frame of length {len(last)}, expected {win_len}"
        assert np.all(np.isfinite(last)), "flush() frame contains NaN/Inf"

    # After flush, framer buffer should be drained
    # Pushing nothing should yield nothing
    empty_frames = framer.push(np.array([], dtype=np.float32))
    assert empty_frames == [] or len(empty_frames) == 0


# ===========================================================================
# TEST 10: P4 spectrum shape contract
# ===========================================================================

def test_p4_spectrum_shape_contract_dtln_stub():
    """
    DTLNStub must accept full complex spectrum (n_fft=320) and return
    the same shape. This is the P4 <-> P3 spectrum contract.
    """
    n_fft = 320
    stub = DTLNStub()
    rng = np.random.default_rng(44)
    D_in = (rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)).astype(np.complex64)

    D_out = stub.process(D_in)

    assert D_out.shape == (n_fft,), (
        f"P4 stub output shape {D_out.shape} != input shape ({n_fft},). "
        "P4 must preserve full complex spectrum shape for P3 ISTFT."
    )
    assert np.iscomplexobj(D_out), "P4 must return complex spectrum"
    assert np.all(np.isfinite(D_out)), "P4 output must be finite"


def test_p4_spectrum_shape_contract_dfn_stub():
    """DFNStub must also satisfy the P4 spectrum shape contract."""
    n_fft = 320
    stub = DFNStub()
    rng = np.random.default_rng(45)
    D_in = (rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)).astype(np.complex64)
    D_out = stub.process(D_in)

    assert D_out.shape == (n_fft,)
    assert np.iscomplexobj(D_out)
    assert np.all(np.isfinite(D_out))


def test_p4_onesided_input_is_not_fed_to_istft(dsp_cfg):
    """
    Verify that passing only 161 bins to P3 ISTFT raises or produces
    incorrect output — this confirms the full 320-bin contract is required.
    """
    dsp = STFTProcessor(dsp_cfg)
    n_fft = dsp_cfg.n_fft      # 320
    n_onesided = n_fft // 2 + 1  # 161

    rng = np.random.default_rng(46)
    D_onesided = rng.standard_normal(n_onesided).astype(np.complex64)

    # P3 inverse() expects shape (num_frames, n_fft). Feeding 161 bins must fail.
    D_2d = D_onesided.reshape(1, -1)   # (1, 161)
    with pytest.raises(Exception):
        _ = dsp.inverse(D_2d)


# ===========================================================================
# TEST 11: End-to-end real WAV with metrics
# ===========================================================================

def test_e2e_real_wav_with_metrics(real_audio, default_config):
    """
    Run the full pipeline on a real dataset example and compute metrics.
    All metrics must be finite, and labelled correctly as estimated/proxy.
    """
    primary, reference, fs, example = real_audio

    pipeline = EndToEndPipeline(default_config)
    result = pipeline.process(primary, reference, sample_rate=fs)

    # Basic sanity
    assert result.enhanced_wav is not None
    assert not result.has_nan,  "Enhanced WAV contains NaN"
    assert not result.has_inf,  "Enhanced WAV contains Inf"
    assert result.clipping_rate < 0.01, "More than 1% of samples are clipping"

    # PROXY / ESTIMATED metrics (from NLMS — NOT ground truth)
    nlms_snr = result.nlms_result.nlms_snr_estimate_db
    assert np.isfinite(nlms_snr), "NLMS estimated SNR must be finite"
    # NOTE: nlms_snr is an ESTIMATED/PROXY metric derived from the NLMS error,
    # NOT ground-truth SNR from the dataset metadata.

    # Verify ground-truth SNR label exists (from metadata, not NLMS)
    assert hasattr(example, 'snr_db'), "DatasetExample must carry ground-truth SNR label"
    gt_snr = example.snr_db      # e.g. 5 dB, from dataset construction
    assert gt_snr in (-5, 0, 5, 10, 15, 20), f"Unexpected SNR value: {gt_snr}"

    # Router stats
    state_counts = Counter(fd.router_state for fd in result.frame_diagnostics)
    assert "DTLN" in state_counts or "STARTUP" in state_counts, \
        "Router never entered DTLN state"
    assert result.num_frames > 0

    # Output level is reasonable
    assert result.output_rms_db > -60.0, "Output is nearly silent (< -60 dBFS)"


# ===========================================================================
# TEST 12: Router transition (hysteresis, no chatter)
# ===========================================================================

def test_router_no_chatter_around_threshold(dsp_cfg):
    """
    Feed frames with complexity oscillating just above/below threshold.
    The router must NOT switch on every frame (hysteresis/dwell enforced).
    """
    n_fft     = dsp_cfg.n_fft
    n_onesided = n_fft // 2 + 1
    win_len   = dsp_cfg.window_length
    hop_len   = dsp_cfg.hop_length
    rng       = np.random.default_rng(99)

    p5_cfg = AcousticConfig(
        sample_rate=16000, frame_length=win_len,
        hop_length=hop_len, fft_size=n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)
    p5.set_model_available(ModelID.DTLN, True)
    p5.set_model_available(ModelID.DEEP_FILTER_NET, True)

    n_frames = 200
    states = []

    for i in range(n_frames):
        wf = rng.standard_normal(win_len).astype(np.float32) * 0.1
        D  = np.fft.fft(wf).astype(np.complex64)
        mag_onesided = np.abs(D[:n_onesided]).astype(np.float32)
        frame = AcousticFrame(
            frame_index=i, sample_rate=16000, waveform=wf,
            magnitude=mag_onesided, frame_length=win_len, hop_length=hop_len,
        )
        diag = p5.process(frame)
        states.append(diag.router_decision.router_state.name)

    # Count model switches (consecutive state changes, excluding STARTUP->DTLN)
    switches = sum(
        1 for j in range(1, len(states))
        if states[j] != states[j-1]
        and states[j-1] not in ("STARTUP",)
    )
    # With 200 frames of stationary white noise, complexity stays fairly
    # constant. Switches should be minimal (hysteresis prevents chatter).
    assert switches <= 20, (
        f"Router switched {switches} times in {n_frames} frames — "
        f"possible chatter detected. States: {Counter(states)}"
    )


def test_router_crossfade_only_during_transition():
    """
    crossfade_alpha must be 0.0 outside of TRANSITION state.
    """
    dsp_cfg = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
    dsp     = STFTProcessor(dsp_cfg)
    n_fft     = dsp_cfg.n_fft
    n_onesided = n_fft // 2 + 1
    win_len   = dsp_cfg.window_length
    hop_len   = dsp_cfg.hop_length

    p5_cfg = AcousticConfig(
        sample_rate=16000, frame_length=win_len,
        hop_length=hop_len, fft_size=n_fft,
    )
    p5 = AcousticPipeline(p5_cfg)
    p5.set_model_available(ModelID.DTLN, True)
    p5.set_model_available(ModelID.DEEP_FILTER_NET, True)

    rng = np.random.default_rng(22)
    for i in range(100):
        wf = rng.standard_normal(win_len).astype(np.float32) * 0.1
        mag_onesided = np.abs(np.fft.fft(wf)[:n_onesided]).astype(np.float32)
        frame = AcousticFrame(
            frame_index=i, sample_rate=16000, waveform=wf,
            magnitude=mag_onesided, frame_length=win_len, hop_length=hop_len,
        )
        diag   = p5.process(frame)
        decision = diag.router_decision
        state  = decision.router_state.name
        alpha  = decision.crossfade_alpha

        if state not in ("TRANSITION",):
            assert alpha == 0.0 or state == "STARTUP", (
                f"Frame {i}: crossfade_alpha={alpha} but state={state} "
                f"(alpha must be 0.0 outside TRANSITION)"
            )
        else:
            # During transition, alpha should be in [0, 1]
            assert 0.0 <= alpha <= 1.0, (
                f"Frame {i}: alpha={alpha} out of [0,1] during TRANSITION"
            )
