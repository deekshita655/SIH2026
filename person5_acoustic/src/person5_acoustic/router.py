"""
router.py
=========
Model Router State Machine for Person 5.

Implements the explicit state machine that decides which speech enhancement
model (DTLN or DeepFilterNet) should be active for a given acoustic frame.

State machine:

    STARTUP
        └── (after warm-up frames) ──────────────────────→ DTLN

    DTLN
        └── (C > T_high for dwell_enter frames) ─────────→ TRANSITION(→DFN)

    DFN
        └── (C < T_low for dwell_exit frames) ───────────→ TRANSITION(→DTLN)

    TRANSITION
        └── (crossfade complete) ────────────────────────→ DTLN or DFN

    FALLBACK
        └── (when target model is unavailable or health fails)

Additional behaviors:
    - If the requested model is unavailable, do NOT route to it.
    - If a model reports a health failure, fall back to DTLN (cheaper).
    - Invalid or NaN complexity values → STAY, no switch.
    - Safe fallback: if all models fail, set state to FALLBACK.

The router does NOT:
    - Implement DTLN or DeepFilterNet
    - Compute audio output
    - Perform FFT or STFT
    - Assume feedback from the model about its own quality

The router DOES:
    - Maintain model availability flags
    - Interact with the HysteresisController and CrossfadeController
    - Produce RouterDecision per frame
    - Expose clean hooks for P4 integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .interfaces import (
    AcousticConfig,
    ModelID,
    RouterState,
    RouterDecision,
)
from .hysteresis import HysteresisController, HysteresisSignal
from .crossfade import CrossfadeController


# ---------------------------------------------------------------------------
# Model Health / Availability
# ---------------------------------------------------------------------------

@dataclass
class ModelHealth:
    """
    Tracks the availability and health of a single model.

    `available`: set externally (e.g., model loaded successfully)
    `healthy`:   cleared by router on detected failure (e.g., crash, timeout)
    """
    model_id: ModelID
    available: bool = True
    healthy: bool = True
    failure_count: int = 0

    @property
    def is_usable(self) -> bool:
        return self.available and self.healthy

    def report_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= 3:  # 3 consecutive failures → mark unhealthy
            self.healthy = False

    def reset_health(self) -> None:
        self.failure_count = 0
        self.healthy = True


# ---------------------------------------------------------------------------
# Router Configuration
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Router-specific configuration, derived from AcousticConfig."""
    startup_frames: int = 10        # frames before leaving STARTUP
    default_model: ModelID = ModelID.DTLN
    fallback_model: ModelID = ModelID.DTLN  # if target unavailable, use this

    @classmethod
    def from_config(cls, cfg: AcousticConfig) -> "RouterConfig":
        return cls(
            startup_frames=cfg.startup_frames,
            default_model=ModelID.DTLN,
            fallback_model=ModelID.DTLN,
        )


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Explicit state machine routing DTLN ↔ DeepFilterNet.

    Integrates:
        - HysteresisController: two-threshold, dwell-time logic
        - CrossfadeController:  alpha ramp for smooth transitions

    State transitions:
        STARTUP   → DTLN        : after startup_frames
        DTLN      → TRANSITION  : C > T_high for dwell_enter frames
        DFN       → TRANSITION  : C < T_low for dwell_exit frames
        TRANSITION → DTLN/DFN   : after crossfade complete
        any        → FALLBACK   : if all models fail

    Usage:
        router = ModelRouter(config)
        router.set_model_available(ModelID.DEEP_FILTER_NET, True)
        decision = router.process(complexity_score, frame_index)

    Args:
        config: AcousticConfig
    """

    def __init__(self, config: AcousticConfig):
        self._config = config
        self._router_cfg = RouterConfig.from_config(config)

        self._hysteresis = HysteresisController(
            threshold_low=config.threshold_low,
            threshold_high=config.threshold_high,
            dwell_enter=config.dwell_enter,
            dwell_exit=config.dwell_exit,
            initial_model=self._router_cfg.default_model,
        )
        self._crossfade = CrossfadeController(
            transition_frames=config.crossfade_frames,
        )

        self._state: RouterState = RouterState.STARTUP
        self._active_model: ModelID = self._router_cfg.default_model
        self._requested_model: ModelID = self._router_cfg.default_model
        self._target_model: Optional[ModelID] = None
        self._frame_count: int = 0

        # Model health tracking
        self._health: dict[ModelID, ModelHealth] = {
            ModelID.DTLN: ModelHealth(ModelID.DTLN, available=True),
            ModelID.DEEP_FILTER_NET: ModelHealth(
                ModelID.DEEP_FILTER_NET, available=False  # unavailable until P3 provides it
            ),
            ModelID.NONE: ModelHealth(ModelID.NONE, available=True),
        }

    def reset(self) -> None:
        """Full reset to startup state."""
        self._hysteresis.reset(self._router_cfg.default_model)
        self._crossfade.reset()
        self._state = RouterState.STARTUP
        self._active_model = self._router_cfg.default_model
        self._requested_model = self._router_cfg.default_model
        self._target_model = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Model availability API (called externally, e.g., by P3/P4)
    # ------------------------------------------------------------------

    def set_model_available(self, model: ModelID, available: bool) -> None:
        """Mark a model as available or unavailable for routing."""
        if model in self._health:
            self._health[model].available = available

    def report_model_failure(self, model: ModelID) -> None:
        """Report that a model produced a failure (e.g., crash, timeout)."""
        if model in self._health:
            self._health[model].report_failure()
            if not self._health[model].is_usable:
                # If active model failed, trigger fallback
                if model == self._active_model:
                    self._trigger_fallback()

    def reset_model_health(self, model: ModelID) -> None:
        """Reset health status (e.g., after model reload)."""
        if model in self._health:
            self._health[model].reset_health()

    def model_availability(self) -> dict[str, bool]:
        """Return {model_name: is_usable} for diagnostics."""
        return {
            m.value: h.is_usable
            for m, h in self._health.items()
        }

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(
        self,
        complexity: float,
        frame_index: int,
    ) -> RouterDecision:
        """
        Process one complexity score and advance the router state machine.

        Args:
            complexity: C(t) ∈ [0, 1] from the complexity engine
            frame_index: monotonically increasing frame counter

        Returns:
            RouterDecision with active model, crossfade alpha, state, etc.
        """
        self._frame_count += 1

        # Validate complexity
        import math
        if math.isnan(complexity) or math.isinf(complexity):
            return self._make_decision(
                frame_index, complexity,
                override_state=self._state,
            )

        # ----------------------------------------------------------------
        # State machine
        # ----------------------------------------------------------------
        if self._state == RouterState.STARTUP:
            self._handle_startup()

        elif self._state == RouterState.DTLN:
            self._handle_steady_state(complexity)

        elif self._state == RouterState.DFN:
            self._handle_steady_state(complexity)

        elif self._state == RouterState.TRANSITION:
            self._handle_transition()

        elif self._state == RouterState.FALLBACK:
            self._handle_fallback(complexity)

        return self._make_decision(frame_index, complexity)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_startup(self) -> None:
        """Leave startup after warming up enough frames."""
        if self._frame_count >= self._router_cfg.startup_frames:
            self._state = RouterState.DTLN
            self._active_model = ModelID.DTLN
            self._requested_model = ModelID.DTLN

    def _handle_steady_state(self, complexity: float) -> None:
        """Handle DTLN or DFN steady states using hysteresis."""
        signal, _ = self._hysteresis.update(complexity)

        if signal == HysteresisSignal.REQUEST_DFN:
            target = ModelID.DEEP_FILTER_NET
            if self._health[target].is_usable:
                self._start_transition(self._active_model, target)
            else:
                # Target unavailable → stay with current model, reset dwell
                self._hysteresis.reset(self._active_model)

        elif signal == HysteresisSignal.REQUEST_DTLN:
            target = ModelID.DTLN
            if self._health[target].is_usable:
                self._start_transition(self._active_model, target)
            else:
                self._trigger_fallback()

        # HysteresisSignal.STAY → do nothing

    def _handle_transition(self) -> None:
        """Advance crossfade; complete transition when ramp finishes."""
        if self._crossfade.is_complete:
            new_model = self._target_model or self._router_cfg.fallback_model
            self._active_model = new_model
            self._requested_model = new_model
            self._target_model = None
            self._state = (
                RouterState.DTLN
                if new_model == ModelID.DTLN
                else RouterState.DFN
            )
            self._hysteresis.notify_switch_complete(new_model)
            self._crossfade.finish()
        else:
            # Advance crossfade one step
            self._crossfade.step()

    def _handle_fallback(self, complexity: float) -> None:
        """
        In fallback state: try to recover.
        If DTLN becomes available, switch back to it.
        """
        if self._health[ModelID.DTLN].is_usable:
            self._active_model = ModelID.DTLN
            self._requested_model = ModelID.DTLN
            self._state = RouterState.DTLN
            self._hysteresis.reset(ModelID.DTLN)

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _start_transition(self, from_model: ModelID, to_model: ModelID) -> None:
        """Begin a crossfade transition."""
        self._target_model = to_model
        self._requested_model = to_model
        self._state = RouterState.TRANSITION
        try:
            self._crossfade.start_transition(from_model, to_model)
        except RuntimeError:
            # Already transitioning — this shouldn't happen, but handle gracefully
            pass

    def _trigger_fallback(self) -> None:
        """Enter fallback state."""
        fallback = self._router_cfg.fallback_model
        self._active_model = fallback
        self._requested_model = fallback
        self._state = RouterState.FALLBACK
        self._crossfade.reset()
        self._target_model = None

    # ------------------------------------------------------------------
    # Decision assembly
    # ------------------------------------------------------------------

    def _make_decision(
        self,
        frame_index: int,
        complexity: float,
        override_state: Optional[RouterState] = None,
    ) -> RouterDecision:
        """Assemble a RouterDecision from current state."""
        alpha = self._crossfade.alpha if self._crossfade.is_transitioning else 0.0

        return RouterDecision(
            frame_index=frame_index,
            active_model=self._active_model,
            requested_model=self._requested_model,
            router_state=override_state or self._state,
            crossfade_alpha=alpha,
            target_model=self._target_model,
            complexity_score=complexity,
            threshold_high=self._hysteresis.threshold_high,
            threshold_low=self._hysteresis.threshold_low,
            dwell_counter=self._hysteresis.dwell_counter,
            transition_frame=self._crossfade.transition_frame,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RouterState:
        return self._state

    @property
    def active_model(self) -> ModelID:
        return self._active_model

    @property
    def crossfade_alpha(self) -> float:
        return self._crossfade.alpha

    @property
    def frame_count(self) -> int:
        return self._frame_count
