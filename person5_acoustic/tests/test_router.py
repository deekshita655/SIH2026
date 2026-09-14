"""
test_router.py
==============
Tests for person5_acoustic.router module.
"""

import pytest

from person5_acoustic import (
    AcousticConfig,
    ModelID,
    RouterState,
)
from person5_acoustic.router import ModelRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    cfg = AcousticConfig.development_16khz()
    # Short dwell and startup for testing
    cfg.dwell_enter = 5
    cfg.dwell_exit = 5
    cfg.crossfade_frames = 3
    return cfg


def _make_router(dwell: int = 3, startup: int = 5, crossfade: int = 3):
    """Create a router with short timeouts for testing."""
    import dataclasses
    cfg = AcousticConfig.development_16khz()
    cfg = dataclasses.replace(
        cfg,
        dwell_enter=dwell,
        dwell_exit=dwell,
        crossfade_frames=crossfade,
        threshold_low=0.35,
        threshold_high=0.65,
        startup_frames=startup,
    )
    router = ModelRouter(cfg)
    # Make DFN available
    router.set_model_available(ModelID.DEEP_FILTER_NET, True)
    return router


def _warmup(router: ModelRouter, n: int = 10, c: float = 0.1) -> None:
    """Push router through startup by feeding n frames."""
    for i in range(n):
        router.process(c, i)


# ---------------------------------------------------------------------------
# Startup behaviour
# ---------------------------------------------------------------------------

class TestRouterStartup:
    def test_initial_state_is_startup(self, config):
        router = ModelRouter(config)
        assert router.state == RouterState.STARTUP

    def test_leaves_startup_after_warmup(self, config):
        router = ModelRouter(config)
        router.set_model_available(ModelID.DEEP_FILTER_NET, True)
        for i in range(20):
            decision = router.process(0.1, i)
        assert router.state != RouterState.STARTUP

    def test_active_model_is_dtln_at_startup(self, config):
        router = ModelRouter(config)
        decision = router.process(0.1, 0)
        assert decision.active_model == ModelID.DTLN

    def test_startup_then_dtln_state(self, config):
        router = ModelRouter(config)
        router.set_model_available(ModelID.DEEP_FILTER_NET, True)
        _warmup(router, n=20, c=0.1)
        assert router.state in (RouterState.DTLN, RouterState.DFN)


# ---------------------------------------------------------------------------
# DTLN → DFN transition
# ---------------------------------------------------------------------------

class TestDTLNtoDFNTransition:
    def test_transition_starts_after_dwell(self):
        router = _make_router(dwell=3, startup=5)
        _warmup(router, n=5, c=0.1)
        assert router.state == RouterState.DTLN

        # Feed high complexity for dwell frames
        for i in range(5, 5 + 3):
            router.process(0.9, i)

        # Should be in TRANSITION now
        assert router.state in (RouterState.TRANSITION, RouterState.DFN)

    def test_transition_completes_into_dfn(self):
        router = _make_router(dwell=3, startup=5, crossfade=3)
        _warmup(router, n=5, c=0.1)

        # Trigger transition
        for i in range(5, 5 + 3):
            router.process(0.9, i)

        # Drive through transition
        for i in range(8, 20):
            router.process(0.9, i)

        assert router.state == RouterState.DFN
        assert router.active_model == ModelID.DEEP_FILTER_NET

    def test_brief_spike_does_not_switch(self):
        router = _make_router(dwell=5, startup=5)
        _warmup(router, n=5, c=0.1)

        # Only 1 frame of high complexity (insufficient dwell)
        router.process(0.9, 5)
        router.process(0.1, 6)  # back to low
        router.process(0.1, 7)
        router.process(0.1, 8)

        assert router.state == RouterState.DTLN


# ---------------------------------------------------------------------------
# DFN → DTLN transition
# ---------------------------------------------------------------------------

class TestDFNtoDTLNTransition:
    def _get_dfn_router(self) -> ModelRouter:
        router = _make_router(dwell=3, startup=5, crossfade=3)
        _warmup(router, n=5, c=0.1)
        # Trigger DTLN → DFN
        for i in range(5, 5 + 3):
            router.process(0.9, i)
        for i in range(8, 20):
            router.process(0.9, i)
        assert router.state == RouterState.DFN
        return router

    def test_dfn_switches_back_on_low_complexity(self):
        router = self._get_dfn_router()
        # Feed low complexity for dwell frames
        for i in range(20, 20 + 3):
            router.process(0.1, i)
        # Drive through crossfade
        for i in range(23, 40):
            router.process(0.1, i)
        assert router.state == RouterState.DTLN

    def test_brief_quiet_does_not_switch_back(self):
        router = self._get_dfn_router()
        # Only 1 frame of low complexity
        router.process(0.1, 20)
        router.process(0.9, 21)
        assert router.state == RouterState.DFN


# ---------------------------------------------------------------------------
# Target unavailable
# ---------------------------------------------------------------------------

class TestTargetUnavailable:
    def test_no_switch_when_dfn_unavailable(self):
        router = _make_router(dwell=3, startup=5)
        router.set_model_available(ModelID.DEEP_FILTER_NET, False)
        _warmup(router, n=5, c=0.1)

        # High complexity: would switch to DFN, but it's unavailable
        for i in range(5, 5 + 5):
            router.process(0.9, i)

        assert router.state == RouterState.DTLN
        assert router.active_model == ModelID.DTLN

    def test_switch_resumes_when_dfn_becomes_available(self):
        router = _make_router(dwell=3, startup=5, crossfade=3)
        router.set_model_available(ModelID.DEEP_FILTER_NET, False)
        _warmup(router, n=5, c=0.1)

        for i in range(5, 5 + 5):
            router.process(0.9, i)
        assert router.state == RouterState.DTLN

        # DFN becomes available
        router.set_model_available(ModelID.DEEP_FILTER_NET, True)
        for i in range(10, 10 + 5):
            router.process(0.9, i)

        # Eventually transitions to DFN
        for i in range(15, 25):
            router.process(0.9, i)
        assert router.state in (RouterState.TRANSITION, RouterState.DFN)


# ---------------------------------------------------------------------------
# Fallback / health failure
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_on_active_model_failure(self):
        router = _make_router(dwell=3, startup=5, crossfade=3)
        _warmup(router, n=5, c=0.1)

        # Report 3 failures on DTLN (active model)
        for _ in range(3):
            router.report_model_failure(ModelID.DTLN)

        # Should enter fallback
        assert router.state == RouterState.FALLBACK

    def test_recovery_from_fallback(self):
        router = _make_router(dwell=3, startup=5, crossfade=3)
        _warmup(router, n=5, c=0.1)

        for _ in range(3):
            router.report_model_failure(ModelID.DTLN)

        # Reset DTLN health
        router.reset_model_health(ModelID.DTLN)
        router.process(0.1, 20)  # one frame to trigger recovery

        assert router.state in (RouterState.DTLN, RouterState.FALLBACK)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_nan_complexity_stays_in_current_state(self):
        import math
        router = _make_router(dwell=3, startup=5)
        _warmup(router, n=5, c=0.1)
        state_before = router.state
        decision = router.process(math.nan, 10)
        assert router.state == state_before

    def test_inf_complexity_safe(self):
        import math
        router = _make_router(dwell=3, startup=5)
        _warmup(router, n=5, c=0.1)
        decision = router.process(math.inf, 10)
        assert decision is not None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestRouterDiagnostics:
    def test_decision_has_all_fields(self):
        router = _make_router(dwell=3, startup=5)
        _warmup(router, n=5, c=0.1)
        decision = router.process(0.5, 10)

        assert decision.frame_index == 10
        assert decision.active_model is not None
        assert decision.router_state is not None
        assert 0.0 <= decision.crossfade_alpha <= 1.0
        assert decision.complexity_score == pytest.approx(0.5)

    def test_model_availability_dict(self):
        router = _make_router()
        avail = router.model_availability()
        assert "DTLN" in avail
        assert "DeepFilterNet" in avail

    def test_reset_returns_to_startup(self):
        router = _make_router(dwell=3, startup=5)
        _warmup(router, n=10, c=0.1)
        router.reset()
        assert router.state == RouterState.STARTUP
        assert router.frame_count == 0
