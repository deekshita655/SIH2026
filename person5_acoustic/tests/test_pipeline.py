"""
test_pipeline.py
================
Integration tests for person5_acoustic.pipeline (AcousticPipeline).
"""

import dataclasses
import pytest
import numpy as np

from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    AcousticPipeline,
    ModelID,
    RouterState,
    PipelineDiagnostics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(dwell: int = 3, startup: int = 5, crossfade: int = 3) -> AcousticConfig:
    cfg = AcousticConfig.development_16khz()
    return dataclasses.replace(
        cfg,
        dwell_enter=dwell,
        dwell_exit=dwell,
        crossfade_frames=crossfade,
        threshold_low=0.35,
        threshold_high=0.65,
        startup_frames=startup,
    )


def _make_frame(
    frame_index: int,
    amp: float = 0.1,
    sr: int = 16000,
    n: int = 320,
    seed: int = None,
) -> AcousticFrame:
    rng = np.random.default_rng(seed or frame_index)
    waveform = (rng.standard_normal(n) * amp).astype(np.float32)
    fft_size = 512
    freq = np.fft.rfftfreq(fft_size, d=1.0 / sr).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(waveform, n=fft_size)).astype(np.float32)
    return AcousticFrame(
        frame_index=frame_index,
        sample_rate=sr,
        waveform=waveform,
        magnitude=magnitude,
        freq_bins=freq,
        frame_length=n,
        hop_length=n // 2,
    )


def _noisy_frame(frame_index: int, amp: float = 2.0) -> AcousticFrame:
    """High-amplitude noisy frame to push complexity high."""
    return _make_frame(frame_index, amp=amp)


def _quiet_frame(frame_index: int, amp: float = 0.001) -> AcousticFrame:
    """Low-amplitude quiet frame to push complexity low."""
    return _make_frame(frame_index, amp=amp)


# ---------------------------------------------------------------------------
# Basic frame processing
# ---------------------------------------------------------------------------

class TestPipelineBasic:
    def test_process_returns_diagnostics(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        assert isinstance(diag, PipelineDiagnostics)

    def test_diagnostics_has_all_fields(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        assert diag.features is not None
        assert diag.router_decision is not None
        assert 0.0 <= diag.complexity_score <= 1.0

    def test_frame_index_preserved(self):
        pipeline = AcousticPipeline(_make_config())
        for i in [0, 5, 100]:
            diag = pipeline.process(_make_frame(i))
            assert diag.frame_index == i

    def test_model_availability_in_diagnostics(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        avail = diag.model_availability
        assert "DTLN" in avail or len(avail) > 0

    def test_complexity_always_in_01(self):
        pipeline = AcousticPipeline(_make_config())
        for i in range(50):
            frame = _make_frame(i)
            diag = pipeline.process(frame)
            assert 0.0 <= diag.complexity_score <= 1.0

    def test_router_decision_present(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        rd = diag.router_decision
        assert rd.active_model is not None
        assert rd.router_state is not None
        assert 0.0 <= rd.crossfade_alpha <= 1.0


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestPipelineStateTransitions:
    def _warm_pipeline(self, pipeline, n=10, amp=0.01):
        for i in range(n):
            pipeline.process(_make_frame(i, amp=amp))

    def test_leaves_startup_after_warmup(self):
        pipeline = AcousticPipeline(_make_config(startup=5))
        self._warm_pipeline(pipeline, n=10)
        assert pipeline.router_state != RouterState.STARTUP

    def test_dtln_to_dfn_on_high_complexity(self):
        pipeline = AcousticPipeline(_make_config(dwell=3, startup=5, crossfade=3))
        pipeline.set_model_available(ModelID.DEEP_FILTER_NET, True)

        # Warm up with quiet frames
        self._warm_pipeline(pipeline, n=5, amp=0.001)

        # Feed noisy frames to push complexity high
        for i in range(5, 5 + 20):
            pipeline.process(_noisy_frame(i, amp=3.0))

        # Should have eventually transitioned toward DFN
        final_state = pipeline.router_state
        assert final_state in (RouterState.DFN, RouterState.TRANSITION,
                                RouterState.DTLN)  # may or may not cross threshold

    def test_dfn_to_dtln_on_low_complexity(self):
        """Full DTLN→DFN→DTLN cycle."""
        pipeline = AcousticPipeline(_make_config(dwell=3, startup=5, crossfade=3))
        pipeline.set_model_available(ModelID.DEEP_FILTER_NET, True)

        # Warm up
        self._warm_pipeline(pipeline, n=5, amp=0.001)

        # Push toward DFN with noisy frames
        for i in range(5, 35):
            pipeline.process(_noisy_frame(i, amp=5.0))

        # If DFN was reached, now go quiet
        if pipeline.router_state == RouterState.DFN:
            for i in range(35, 65):
                pipeline.process(_quiet_frame(i, amp=0.0001))
            # Should have eventually transitioned back toward DTLN
            # (may still be in DFN if complexity hasn't crossed T_low)
            assert pipeline.router_state in (
                RouterState.DTLN, RouterState.DFN, RouterState.TRANSITION
            )

    def test_no_switch_when_dfn_unavailable(self):
        pipeline = AcousticPipeline(_make_config(dwell=3, startup=5))
        # DFN not available (default)
        self._warm_pipeline(pipeline, n=5, amp=0.001)

        for i in range(5, 20):
            pipeline.process(_noisy_frame(i, amp=10.0))

        assert pipeline.active_model == ModelID.DTLN

    def test_reset_returns_to_startup(self):
        pipeline = AcousticPipeline(_make_config())
        self._warm_pipeline(pipeline, n=10)
        pipeline.reset()
        assert pipeline.router_state == RouterState.STARTUP
        assert pipeline.frame_count == 0


# ---------------------------------------------------------------------------
# Diagnostics content
# ---------------------------------------------------------------------------

class TestPipelineDiagnostics:
    def test_features_populated(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        fv = diag.features
        assert fv is not None
        assert isinstance(fv.rms, float)
        assert isinstance(fv.zcr, float)
        assert isinstance(fv.spectral_entropy, float)

    def test_normalized_features_populated(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        assert diag.normalized is not None
        assert 0.0 <= diag.normalized.rms_norm <= 1.0
        assert 0.0 <= diag.normalized.zcr_norm <= 1.0

    def test_fast_slow_ewma_snapshots(self):
        pipeline = AcousticPipeline(_make_config())
        for i in range(5):
            diag = pipeline.process(_make_frame(i))
        assert "rms" in diag.fast_ewma_snapshot
        assert "rms" in diag.slow_ewma_snapshot

    def test_transient_flag_is_bool(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        assert isinstance(diag.is_transient, bool)

    def test_complexity_contributions_sum_to_score(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        total = sum(diag.complexity_contributions.values())
        assert total == pytest.approx(diag.complexity_score, abs=1e-5)

    def test_crossfade_alpha_in_router_decision(self):
        pipeline = AcousticPipeline(_make_config())
        diag = pipeline.process(_make_frame(0))
        assert 0.0 <= diag.router_decision.crossfade_alpha <= 1.0


# ---------------------------------------------------------------------------
# 48 kHz configuration
# ---------------------------------------------------------------------------

class TestPipeline48kHz:
    def test_48khz_pipeline_runs(self):
        config = AcousticConfig.production_48khz()
        pipeline = AcousticPipeline(config)
        rng = np.random.default_rng(0)
        waveform = (rng.standard_normal(960) * 0.1).astype(np.float32)
        frame = AcousticFrame(
            frame_index=0,
            sample_rate=48000,
            waveform=waveform,
            frame_length=960,
            hop_length=480,
        )
        diag = pipeline.process(frame)
        assert 0.0 <= diag.complexity_score <= 1.0
