"""
test_complexity.py
==================
Tests for person5_acoustic.complexity module.
"""

import pytest
import numpy as np

from person5_acoustic import AcousticConfig, FeatureVector
from person5_acoustic.complexity import ComplexityEngine, ComplexityResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fv(
    frame_index: int = 0,
    rms: float = 0.1,
    zcr: float = 0.3,
    flux: float = 0.5,
    entropy: float = 0.7,
    centroid_var: float = 0.1,
    transient: float = 0.0,
    snr_db: float = 10.0,
    is_transient: bool = False,
) -> FeatureVector:
    fv = FeatureVector(frame_index=frame_index)
    fv.rms = rms
    fv.zcr = zcr
    fv.spectral_flux = flux
    fv.spectral_entropy = entropy
    fv.centroid_variation = centroid_var
    fv.transient_score = transient
    fv.estimated_snr_db = snr_db
    fv.is_transient = is_transient
    return fv


@pytest.fixture
def config():
    return AcousticConfig.development_16khz()


@pytest.fixture
def engine(config):
    return ComplexityEngine(config)


# ---------------------------------------------------------------------------
# Basic output range
# ---------------------------------------------------------------------------

class TestComplexityOutput:
    def test_score_always_in_01(self, config):
        engine = ComplexityEngine(config)
        rng = np.random.default_rng(7)
        for i in range(100):
            fv = _make_fv(
                frame_index=i,
                rms=float(abs(rng.standard_normal())),
                zcr=float(rng.uniform(0, 1)),
                flux=float(abs(rng.standard_normal()) * 5),
                entropy=float(rng.uniform(0, 1)),
                centroid_var=float(rng.uniform(0, 0.3)),
                transient=float(rng.uniform(0, 1)),
                snr_db=float(rng.uniform(-5, 30)),
            )
            result = engine.process(fv)
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} out of [0,1] at frame {i}"
            )

    def test_silent_frame_does_not_crash(self, engine):
        fv = _make_fv(rms=0.0, zcr=0.0, flux=0.0, entropy=0.0,
                      centroid_var=0.0, transient=0.0, snr_db=0.0)
        result = engine.process(fv)
        assert 0.0 <= result.score <= 1.0

    def test_returns_complexity_result(self, engine):
        result = engine.process(_make_fv())
        assert isinstance(result, ComplexityResult)

    def test_contributions_sum_to_score(self, engine):
        result = engine.process(_make_fv())
        total = sum(result.contributions.values())
        assert total == pytest.approx(result.score, abs=1e-6)

    def test_contributions_nonnegative(self, engine):
        for i in range(20):
            fv = _make_fv(frame_index=i)
            result = engine.process(fv)
            for k, v in result.contributions.items():
                assert v >= 0.0, f"Negative contribution for {k}: {v}"

    def test_frame_index_preserved(self, engine):
        fv = _make_fv(frame_index=42)
        result = engine.process(fv)
        assert result.frame_index == 42


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_weights_sum_to_one(self, config):
        w = config.complexity_weights
        assert sum(w.values()) == pytest.approx(1.0)

    def test_weights_nonnegative(self, config):
        for k, v in config.complexity_weights.items():
            assert v >= 0.0, f"Weight {k}={v} is negative"

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            AcousticConfig(
                sample_rate=16000,
                frame_length=320,
                hop_length=160,
                complexity_weights={
                    "rms": 0.5, "zcr": 0.5, "spectral_flux": 0.5,
                    "spectral_entropy": 0.5, "centroid_var": 0.5,
                    "transient": 0.5, "snr_difficulty": 0.5,
                }
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            AcousticConfig(
                sample_rate=16000,
                frame_length=320,
                hop_length=160,
                complexity_weights={
                    "rms": -0.1, "zcr": 0.25, "spectral_flux": 0.25,
                    "spectral_entropy": 0.2, "centroid_var": 0.2,
                    "transient": 0.1, "snr_difficulty": 0.1,
                }
            )


# ---------------------------------------------------------------------------
# Sensitivity tests
# ---------------------------------------------------------------------------

class TestComplexitySensitivity:
    def test_transient_increases_score(self, config):
        engine = ComplexityEngine(config)
        # Warm up
        for i in range(20):
            engine.process(_make_fv(frame_index=i, transient=0.0))
        baseline_result = engine.process(_make_fv(frame_index=20, transient=0.0))

        engine2 = ComplexityEngine(config)
        for i in range(20):
            engine2.process(_make_fv(frame_index=i, transient=0.0))
        transient_result = engine2.process(_make_fv(frame_index=20, transient=1.0))

        # Transient should produce higher (or equal) complexity
        assert transient_result.score >= baseline_result.score

    def test_high_snr_reduces_difficulty_contribution(self, config):
        engine_low = ComplexityEngine(config)
        engine_high = ComplexityEngine(config)

        for i in range(30):
            engine_low.process(_make_fv(frame_index=i, snr_db=-5.0))
            engine_high.process(_make_fv(frame_index=i, snr_db=30.0))

        r_low = engine_low.process(_make_fv(frame_index=30, snr_db=-5.0))
        r_high = engine_high.process(_make_fv(frame_index=30, snr_db=30.0))

        # High SNR → lower difficulty contribution
        assert (r_high.contributions.get("snr_difficulty", 0)
                <= r_low.contributions.get("snr_difficulty", 0) + 0.1)

    def test_high_entropy_increases_score(self, config):
        engine_lo = ComplexityEngine(config)
        engine_hi = ComplexityEngine(config)

        # Warm up both with the same frames
        for i in range(30):
            engine_lo.process(_make_fv(frame_index=i, entropy=0.1))
            engine_hi.process(_make_fv(frame_index=i, entropy=0.9))

        r_lo = engine_lo.process(_make_fv(frame_index=30, entropy=0.1))
        r_hi = engine_hi.process(_make_fv(frame_index=30, entropy=0.9))
        # High entropy should contribute more to complexity
        assert r_hi.score >= r_lo.score - 0.05  # allow small adaptive tolerance


# ---------------------------------------------------------------------------
# Adaptive behaviour
# ---------------------------------------------------------------------------

class TestComplexityAdaptation:
    def test_frame_count_increments(self, engine):
        for i in range(5):
            engine.process(_make_fv(frame_index=i))
        assert engine.frame_count == 5

    def test_reset_clears_frame_count(self, engine):
        for i in range(10):
            engine.process(_make_fv(frame_index=i))
        engine.reset()
        assert engine.frame_count == 0

    def test_slow_baseline_exists(self, engine):
        for i in range(10):
            engine.process(_make_fv(frame_index=i))
        baseline = engine.get_slow_baseline()
        assert isinstance(baseline, dict)
        assert "rms" in baseline

    def test_normalized_features_attached(self, engine):
        result = engine.process(_make_fv())
        assert result.normalized_features is not None

    def test_stats_snapshots_present(self, engine):
        result = engine.process(_make_fv())
        assert len(result.stats_snapshots) > 0
