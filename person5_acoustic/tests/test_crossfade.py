"""
test_crossfade.py
=================
Tests for person5_acoustic.crossfade module.
"""

import math
import pytest

from person5_acoustic import ModelID
from person5_acoustic.crossfade import CrossfadeController, RampType, blend


# ---------------------------------------------------------------------------
# CrossfadeController tests
# ---------------------------------------------------------------------------

class TestCrossfadeController:
    @pytest.fixture
    def cf5(self):
        return CrossfadeController(transition_frames=5)

    @pytest.fixture
    def cf1(self):
        return CrossfadeController(transition_frames=1)

    # -----------------------------------------------------------------------
    # Idle state
    # -----------------------------------------------------------------------

    def test_idle_alpha_is_zero(self, cf5):
        assert cf5.alpha == pytest.approx(0.0)

    def test_idle_not_transitioning(self, cf5):
        assert not cf5.is_transitioning

    def test_idle_not_complete(self, cf5):
        assert not cf5.is_complete

    # -----------------------------------------------------------------------
    # Start transition
    # -----------------------------------------------------------------------

    def test_start_sets_transitioning(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        assert cf5.is_transitioning

    def test_alpha_zero_at_start(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        assert cf5.alpha == pytest.approx(0.0)

    def test_from_and_to_models(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        assert cf5.from_model == ModelID.DTLN
        assert cf5.to_model == ModelID.DEEP_FILTER_NET

    def test_same_model_raises(self, cf5):
        with pytest.raises(ValueError):
            cf5.start_transition(ModelID.DTLN, ModelID.DTLN)

    def test_double_start_raises(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        with pytest.raises(RuntimeError):
            cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)

    # -----------------------------------------------------------------------
    # Alpha ramp — linear
    # -----------------------------------------------------------------------

    def test_linear_ramp_reaches_one(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        alpha = 0.0
        for _ in range(5):
            alpha = cf5.step()
        assert alpha == pytest.approx(1.0)

    def test_linear_ramp_monotone(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        prev = cf5.alpha
        for _ in range(5):
            curr = cf5.step()
            assert curr >= prev
            prev = curr

    def test_linear_ramp_values(self):
        """Check specific linear ramp values for N=4."""
        cf = CrossfadeController(transition_frames=4, ramp_type=RampType.LINEAR)
        cf.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        expected = [0.25, 0.5, 0.75, 1.0]
        for e in expected:
            alpha = cf.step()
            assert alpha == pytest.approx(e)

    # -----------------------------------------------------------------------
    # Alpha ramp — equal power
    # -----------------------------------------------------------------------

    def test_equal_power_reaches_one(self):
        cf = CrossfadeController(transition_frames=5, ramp_type=RampType.EQUAL_POWER)
        cf.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        alpha = 0.0
        for _ in range(5):
            alpha = cf.step()
        assert alpha == pytest.approx(1.0)

    def test_equal_power_bounded(self):
        cf = CrossfadeController(transition_frames=5, ramp_type=RampType.EQUAL_POWER)
        cf.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        for _ in range(5):
            alpha = cf.step()
            assert 0.0 <= alpha <= 1.0

    def test_equal_power_midpoint(self):
        """At midpoint t=0.5, sin²(π/4) = 0.5"""
        cf = CrossfadeController(transition_frames=2, ramp_type=RampType.EQUAL_POWER)
        cf.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        alpha = cf.step()  # frame 1 of 2 → t=0.5
        assert alpha == pytest.approx(0.5, abs=0.01)

    # -----------------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------------

    def test_is_complete_after_full_ramp(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        for _ in range(5):
            cf5.step()
        assert cf5.is_complete

    def test_not_complete_mid_ramp(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        cf5.step()  # only 1 of 5 frames
        assert not cf5.is_complete

    def test_finish_clears_transitioning(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        for _ in range(5):
            cf5.step()
        cf5.finish()
        assert not cf5.is_transitioning

    def test_step_raises_when_not_transitioning(self, cf5):
        with pytest.raises(RuntimeError):
            cf5.step()

    # -----------------------------------------------------------------------
    # Transition length 1
    # -----------------------------------------------------------------------

    def test_single_frame_transition(self, cf1):
        cf1.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        alpha = cf1.step()
        assert alpha == pytest.approx(1.0)
        assert cf1.is_complete

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def test_reset_clears_state(self, cf5):
        cf5.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)
        cf5.step()
        cf5.reset()
        assert not cf5.is_transitioning
        assert cf5.alpha == pytest.approx(0.0)

    def test_invalid_transition_frames_raises(self):
        with pytest.raises(ValueError):
            CrossfadeController(transition_frames=0)


# ---------------------------------------------------------------------------
# Blend utility function
# ---------------------------------------------------------------------------

class TestBlend:
    def test_alpha_zero_returns_old(self):
        result = blend(output_old=2.0, output_new=8.0, alpha=0.0)
        assert result == pytest.approx(2.0)

    def test_alpha_one_returns_new(self):
        result = blend(output_old=2.0, output_new=8.0, alpha=1.0)
        assert result == pytest.approx(8.0)

    def test_alpha_half_interpolates(self):
        result = blend(output_old=0.0, output_new=1.0, alpha=0.5)
        assert result == pytest.approx(0.5)

    def test_alpha_clipped_below_zero(self):
        result = blend(0.0, 1.0, alpha=-0.5)
        assert result == pytest.approx(0.0)

    def test_alpha_clipped_above_one(self):
        result = blend(0.0, 1.0, alpha=2.0)
        assert result == pytest.approx(1.0)

    def test_correct_interpolation_formula(self):
        """S = (1-alpha)*S_old + alpha*S_new"""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = blend(10.0, 20.0, alpha)
            expected = (1.0 - alpha) * 10.0 + alpha * 20.0
            assert result == pytest.approx(expected)
