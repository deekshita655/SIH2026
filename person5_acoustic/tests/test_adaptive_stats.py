"""
test_adaptive_stats.py
======================
Tests for person5_acoustic.adaptive_stats module.
"""

import pytest
import numpy as np

from person5_acoustic.adaptive_stats import (
    RunningMedian,
    RunningMAD,
    OutlierDetector,
    EWMA,
    ThresholdedEWMA,
    FeatureAdaptiveStats,
)


# ---------------------------------------------------------------------------
# RunningMedian tests
# ---------------------------------------------------------------------------

class TestRunningMedian:
    def test_single_value(self):
        rm = RunningMedian(window=5)
        v = rm.update(3.0)
        assert v == pytest.approx(3.0)

    def test_monotone_ascending(self):
        rm = RunningMedian(window=5)
        for i in range(1, 6):
            rm.update(float(i))
        # Window: [1,2,3,4,5], median = 3
        assert rm.value == pytest.approx(3.0)

    def test_window_sliding(self):
        rm = RunningMedian(window=3)
        rm.update(1.0)
        rm.update(2.0)
        rm.update(3.0)
        assert rm.value == pytest.approx(2.0)
        rm.update(10.0)  # evicts 1
        # Window: [2,3,10] → median = 3
        assert rm.value == pytest.approx(3.0)

    def test_window_size_1(self):
        rm = RunningMedian(window=1)
        rm.update(5.0)
        assert rm.value == pytest.approx(5.0)
        rm.update(7.0)
        assert rm.value == pytest.approx(7.0)

    def test_is_warm_when_full(self):
        rm = RunningMedian(window=3)
        assert not rm.is_warm
        rm.update(1.0)
        assert not rm.is_warm
        rm.update(2.0)
        assert not rm.is_warm
        rm.update(3.0)
        assert rm.is_warm

    def test_reset_clears_buffer(self):
        rm = RunningMedian(window=5)
        for i in range(5):
            rm.update(float(i))
        rm.reset()
        assert rm.count == 0
        rm.update(99.0)
        assert rm.value == pytest.approx(99.0)

    def test_median_is_robust_to_outlier(self):
        rm = RunningMedian(window=5)
        for _ in range(4):
            rm.update(1.0)
        rm.update(1000.0)  # single spike
        # Median should be 1.0, not affected by spike
        assert rm.value == pytest.approx(1.0)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            RunningMedian(window=0)

    def test_even_window_median(self):
        rm = RunningMedian(window=4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            rm.update(v)
        # np.median([1,2,3,4]) = 2.5
        assert rm.value == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# RunningMAD tests
# ---------------------------------------------------------------------------

class TestRunningMAD:
    def test_constant_signal_has_zero_mad(self):
        rm = RunningMAD(window=5)
        for _ in range(5):
            rm.update(3.0)
        median, mad = rm.values
        assert median == pytest.approx(3.0)
        assert mad == pytest.approx(0.0)

    def test_uniform_values(self):
        rm = RunningMAD(window=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rm.update(v)
        median, mad = rm.values
        assert median == pytest.approx(3.0)
        # Deviations: [2,1,0,1,2] → median = 1
        assert mad == pytest.approx(1.0)

    def test_outlier_does_not_dominate_mad(self):
        rm = RunningMAD(window=7)
        for _ in range(6):
            rm.update(1.0)
        rm.update(1000.0)  # single spike
        median, mad = rm.values
        # Median should still be 1.0
        assert median == pytest.approx(1.0)
        # MAD: |1-1|*6=0 and |1000-1|=999, median of [0,0,0,0,0,0,999] = 0
        assert mad == pytest.approx(0.0)

    def test_returns_tuple(self):
        rm = RunningMAD(window=3)
        result = rm.update(1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset(self):
        rm = RunningMAD(window=5)
        for v in [1.0, 2.0, 3.0]:
            rm.update(v)
        rm.reset()
        assert rm.count == 0
        median, mad = rm.values
        assert median == pytest.approx(0.0)
        assert mad == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# OutlierDetector tests
# ---------------------------------------------------------------------------

class TestOutlierDetector:
    def test_normal_values_not_outliers(self):
        det = OutlierDetector(window=31, k=3.0)
        # Feed many values from a stable distribution
        for v in np.linspace(0, 1, 31):
            is_out, _, _ = det.update(float(v))
        # The last value (1.0) should not be classified as outlier in this range
        is_out, median, mad = det.update(0.5)
        assert not is_out

    def test_impulse_detected_as_outlier(self):
        det = OutlierDetector(window=31, k=3.0)
        # Warm up with constant signal
        for _ in range(31):
            det.update(0.1)
        # Big spike
        is_out, median, mad = det.update(100.0)
        assert is_out

    def test_mad_near_zero_safety(self):
        det = OutlierDetector(window=31, k=3.0, epsilon=1e-8)
        # All same values → MAD = 0
        for _ in range(31):
            det.update(5.0)
        # Small perturbation: should be outlier since MAD ≈ 0
        is_out, _, _ = det.update(5.01)
        assert is_out

    def test_outlier_does_not_reset_baseline(self):
        det = OutlierDetector(window=31, k=3.0)
        for _ in range(31):
            det.update(1.0)
        det.update(1000.0)  # outlier
        # Subsequent normal value should still not be outlier
        is_out, _, _ = det.update(1.0)
        assert not is_out


# ---------------------------------------------------------------------------
# EWMA tests
# ---------------------------------------------------------------------------

class TestEWMA:
    def test_initialization(self):
        e = EWMA(alpha=0.1)
        assert not e.is_initialized
        v = e.update(5.0)
        assert e.is_initialized
        assert v == pytest.approx(5.0)

    def test_converges_to_constant_input(self):
        e = EWMA(alpha=0.1)
        for _ in range(500):
            e.update(10.0)
        assert e.value == pytest.approx(10.0, abs=0.01)

    def test_exponential_decay(self):
        # After 1 update with alpha=0.5 starting at 0: value = 0.5*1 = 0.5
        e = EWMA(alpha=0.5, initial=0.0)
        v = e.update(1.0)
        assert v == pytest.approx(0.5)

    def test_reset_clears_value(self):
        e = EWMA(alpha=0.1)
        e.update(100.0)
        e.reset()
        assert not e.is_initialized

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            EWMA(alpha=0.0)
        with pytest.raises(ValueError):
            EWMA(alpha=1.1)

    def test_alpha_1_equals_last_value(self):
        e = EWMA(alpha=1.0, initial=0.0)
        v = e.update(42.0)
        assert v == pytest.approx(42.0)

    def test_tracks_step_change(self):
        e = EWMA(alpha=0.5)
        e.update(0.0)
        e.update(0.0)
        e.update(1.0)  # step change
        # After 1 update on step: 0.5*0 + 0.5*1 = 0.5
        assert e.value > 0.0
        assert e.value < 1.0


# ---------------------------------------------------------------------------
# ThresholdedEWMA tests
# ---------------------------------------------------------------------------

class TestThresholdedEWMA:
    def test_accepts_normal_samples(self):
        tewma = ThresholdedEWMA(alpha=0.1, window=5, k=3.0)
        for v in [1.0, 1.1, 0.9, 1.05, 0.95]:
            val, is_out = tewma.update(float(v))
        # After 5 samples, EWMA should be close to 1.0
        assert tewma.value == pytest.approx(1.0, abs=0.5)

    def test_rejects_outlier_from_ewma(self):
        tewma = ThresholdedEWMA(alpha=0.1, window=31, k=3.0)
        # Build stable baseline
        for _ in range(31):
            tewma.update(1.0)
        baseline_before = tewma.value
        # Insert huge outlier
        _, is_out = tewma.update(1000.0)
        assert is_out
        # EWMA should NOT have moved much
        assert abs(tewma.value - baseline_before) < 0.5

    def test_accepted_and_rejected_counts(self):
        tewma = ThresholdedEWMA(alpha=0.1, window=31, k=3.0)
        # All normal values
        for _ in range(31):
            tewma.update(1.0)
        # First sample always accepted
        assert tewma.accepted_count > 0
        # Now insert outlier
        tewma.update(1000.0)
        assert tewma.rejected_count >= 1

    def test_initialization_on_first_sample(self):
        tewma = ThresholdedEWMA(alpha=0.1, window=5, k=3.0)
        val, _ = tewma.update(5.0)
        assert tewma.is_initialized
        assert val == pytest.approx(5.0)

    def test_reset_clears_all_state(self):
        tewma = ThresholdedEWMA(alpha=0.1, window=5, k=3.0)
        for v in range(10):
            tewma.update(float(v))
        tewma.reset()
        assert not tewma.is_initialized
        assert tewma.accepted_count == 0
        assert tewma.rejected_count == 0


# ---------------------------------------------------------------------------
# FeatureAdaptiveStats tests
# ---------------------------------------------------------------------------

class TestFeatureAdaptiveStats:
    def test_returns_snapshot(self):
        fas = FeatureAdaptiveStats(fast_alpha=0.1, slow_alpha=0.01, window=5)
        snap = fas.update(1.0)
        assert snap.fast_ewma is not None
        assert snap.slow_ewma is not None
        assert isinstance(snap.is_outlier, bool)

    def test_slow_ewma_slower_than_fast(self):
        fas = FeatureAdaptiveStats(fast_alpha=0.5, slow_alpha=0.01, window=5)
        # Warm up at 1.0
        for _ in range(10):
            fas.update(1.0)
        # Step change to 0.0
        for _ in range(5):
            snap = fas.update(0.0)
        # Fast EWMA should have dropped more than slow
        assert snap.fast_ewma < snap.slow_ewma

    def test_outlier_detected_correctly(self):
        fas = FeatureAdaptiveStats(fast_alpha=0.1, slow_alpha=0.01, window=31)
        for _ in range(31):
            fas.update(1.0)
        snap = fas.update(100.0)
        assert snap.is_outlier

    def test_reset(self):
        fas = FeatureAdaptiveStats(fast_alpha=0.1, slow_alpha=0.01, window=5)
        for v in [1.0, 2.0, 3.0]:
            fas.update(v)
        fas.reset()
        snap = fas.update(5.0)
        assert snap.median == pytest.approx(5.0)
