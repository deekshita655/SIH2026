"""
test_hysteresis.py
==================
Tests for person5_acoustic.hysteresis module.
"""

import pytest

from person5_acoustic import ModelID
from person5_acoustic.hysteresis import (
    HysteresisController,
    HysteresisSignal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def controller():
    """Default hysteresis controller: T_low=0.35, T_high=0.65, dwell=5."""
    return HysteresisController(
        threshold_low=0.35,
        threshold_high=0.65,
        dwell_enter=5,
        dwell_exit=5,
        initial_model=ModelID.DTLN,
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestHysteresisConstruction:
    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            HysteresisController(
                threshold_low=0.7, threshold_high=0.5,
                dwell_enter=5, dwell_exit=5,
            )

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError):
            HysteresisController(
                threshold_low=0.5, threshold_high=0.5,
            )

    def test_valid_construction(self):
        hc = HysteresisController(0.3, 0.7, 5, 5)
        assert hc.threshold_low == pytest.approx(0.3)
        assert hc.threshold_high == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Boundary / steady-state behaviour
# ---------------------------------------------------------------------------

class TestHysteresisBoundary:
    def test_stay_in_hysteresis_zone(self, controller):
        """Complexity in [T_low, T_high] → always STAY."""
        for c in [0.35, 0.5, 0.64, 0.40]:
            signal, _ = controller.update(c)
            assert signal == HysteresisSignal.STAY

    def test_dtln_zone_below_t_low_no_switch_toward_dtln(self, controller):
        """From DTLN, complexity < T_low should NOT trigger switch (already in DTLN)."""
        for _ in range(10):
            signal, _ = controller.update(0.1)
        # Should not be requesting DTLN (we're already there)
        assert signal != HysteresisSignal.REQUEST_DTLN

    def test_dtln_above_threshold_does_not_immediately_switch(self, controller):
        """Need dwell_enter frames above T_high before requesting DFN."""
        # Only 1 frame above T_high
        signal, _ = controller.update(0.9)
        assert signal == HysteresisSignal.STAY

    def test_dtln_persistent_high_requests_dfn(self, controller):
        """After dwell_enter=5 consecutive frames above T_high, request DFN."""
        for _ in range(5):
            signal, dwell = controller.update(0.9)
        assert signal == HysteresisSignal.REQUEST_DFN

    def test_dwell_counter_resets_on_return_to_zone(self, controller):
        """If complexity drops back into zone before dwell, counter resets."""
        for _ in range(4):
            controller.update(0.9)  # 4 frames above T_high
        controller.update(0.5)      # drops back to zone → reset
        signal, dwell = controller.update(0.9)  # start counting again
        assert dwell == 1  # counter reset


# ---------------------------------------------------------------------------
# DFN → DTLN switching
# ---------------------------------------------------------------------------

class TestDFNtoTDLNSwitching:
    @pytest.fixture
    def dfn_controller(self):
        hc = HysteresisController(
            threshold_low=0.35,
            threshold_high=0.65,
            dwell_enter=5,
            dwell_exit=5,
            initial_model=ModelID.DEEP_FILTER_NET,
        )
        return hc

    def test_dfn_below_low_requests_dtln_after_dwell(self, dfn_controller):
        for _ in range(5):
            signal, _ = dfn_controller.update(0.1)
        assert signal == HysteresisSignal.REQUEST_DTLN

    def test_dfn_above_high_no_switch(self, dfn_controller):
        """From DFN, complexity above T_high should NOT trigger switch."""
        for _ in range(10):
            signal, _ = dfn_controller.update(0.9)
        assert signal != HysteresisSignal.REQUEST_DFN  # already in DFN


# ---------------------------------------------------------------------------
# No chatter around threshold
# ---------------------------------------------------------------------------

class TestNoChatter:
    def test_alternating_around_threshold_no_switch(self):
        """Alternating above/below T_high → never meets dwell requirement."""
        hc = HysteresisController(0.35, 0.65, dwell_enter=5, dwell_exit=5)
        switches = 0
        for i in range(100):
            c = 0.9 if i % 2 == 0 else 0.5  # oscillate
            signal, _ = hc.update(c)
            if signal == HysteresisSignal.REQUEST_DFN:
                switches += 1
        assert switches == 0

    def test_consistent_signal_does_eventually_switch(self):
        """Persistent signal above T_high eventually produces REQUEST_DFN."""
        hc = HysteresisController(0.35, 0.65, dwell_enter=5, dwell_exit=5)
        signals = [hc.update(0.9)[0] for _ in range(10)]
        assert HysteresisSignal.REQUEST_DFN in signals


# ---------------------------------------------------------------------------
# Notify switch complete
# ---------------------------------------------------------------------------

class TestNotifySwitchComplete:
    def test_notify_updates_current_model(self, controller):
        controller.notify_switch_complete(ModelID.DEEP_FILTER_NET)
        assert controller.current_model == ModelID.DEEP_FILTER_NET

    def test_notify_resets_dwell_counter(self, controller):
        for _ in range(4):
            controller.update(0.9)
        controller.notify_switch_complete(ModelID.DEEP_FILTER_NET)
        assert controller.dwell_counter == 0


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_complexity_above_1_returns_stay(self, controller):
        signal, _ = controller.update(1.5)
        assert signal == HysteresisSignal.STAY

    def test_complexity_below_0_returns_stay(self, controller):
        signal, _ = controller.update(-0.1)
        assert signal == HysteresisSignal.STAY


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestHysteresisReset:
    def test_reset_clears_dwell(self, controller):
        for _ in range(4):
            controller.update(0.9)
        controller.reset(ModelID.DTLN)
        assert controller.dwell_counter == 0

    def test_reset_restores_model(self, controller):
        controller.notify_switch_complete(ModelID.DEEP_FILTER_NET)
        controller.reset(ModelID.DTLN)
        assert controller.current_model == ModelID.DTLN
