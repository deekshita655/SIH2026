"""
hysteresis.py
=============
Hysteresis controller and dwell-time logic for Person 5.

Implements the two-threshold hysteresis system that prevents rapid model
switching when the complexity score fluctuates near a single threshold.

    If current model is DTLN:
        switch toward DFN only if C(t) > T_high

    If current model is DFN:
        switch toward DTLN only if C(t) < T_low

    If T_low <= C(t) <= T_high:
        keep current model (hysteresis zone)

    T_low < T_high  (enforced at construction)

Dwell-time logic:
    The threshold condition must persist for N consecutive frames before
    a switch is requested. This prevents single-frame spikes from
    triggering transitions.

    - dwell_enter: frames required to switch INTO the higher-cost model (DFN)
    - dwell_exit:  frames required to switch BACK to the cheaper model (DTLN)

These are configurable and should be validated in development.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

from .interfaces import AcousticConfig, ModelID


# ---------------------------------------------------------------------------
# Hysteresis direction enum
# ---------------------------------------------------------------------------

class HysteresisSignal(Enum):
    """Output signal from the hysteresis controller."""
    STAY = auto()          # remain with current model
    REQUEST_DFN = auto()   # request switch to DeepFilterNet
    REQUEST_DTLN = auto()  # request switch to DTLN


# ---------------------------------------------------------------------------
# Hysteresis Controller
# ---------------------------------------------------------------------------

class HysteresisController:
    """
    Two-threshold hysteresis controller.

    State:
        - current_model:    which model is currently considered active
        - dwell_counter:    consecutive frames the threshold condition has held
        - pending_signal:   what kind of switch is pending (or None)

    The controller does NOT interact with the actual models. It only
    produces a HysteresisSignal that the router can act on.

    Args:
        threshold_low:  complexity below this → request DTLN
        threshold_high: complexity above this → request DFN
        dwell_enter:    frames to persist above T_high before requesting DFN
        dwell_exit:     frames to persist below T_low before requesting DTLN
        initial_model:  model assumed at startup
    """

    def __init__(
        self,
        threshold_low: float,
        threshold_high: float,
        dwell_enter: int = 5,
        dwell_exit: int = 5,
        initial_model: ModelID = ModelID.DTLN,
    ):
        if threshold_low >= threshold_high:
            raise ValueError(
                f"threshold_low ({threshold_low}) must be < "
                f"threshold_high ({threshold_high})"
            )
        self._t_low = threshold_low
        self._t_high = threshold_high
        self._dwell_enter = max(1, dwell_enter)
        self._dwell_exit = max(1, dwell_exit)
        self._current_model = initial_model

        self._dwell_counter: int = 0
        self._pending_direction: Optional[str] = None  # "up" or "down" or None

    def reset(self, model: ModelID = ModelID.DTLN) -> None:
        """Reset to initial state."""
        self._current_model = model
        self._dwell_counter = 0
        self._pending_direction = None

    @property
    def current_model(self) -> ModelID:
        return self._current_model

    @property
    def dwell_counter(self) -> int:
        return self._dwell_counter

    @property
    def threshold_low(self) -> float:
        return self._t_low

    @property
    def threshold_high(self) -> float:
        return self._t_high

    def notify_switch_complete(self, new_model: ModelID) -> None:
        """
        Called by the router when a model transition is complete.

        This keeps the hysteresis controller in sync with the actual
        active model so it correctly uses the right threshold for the
        next decision.
        """
        self._current_model = new_model
        self._dwell_counter = 0
        self._pending_direction = None

    def update(self, complexity: float) -> tuple[HysteresisSignal, int]:
        """
        Process one complexity sample and return the hysteresis decision.

        Args:
            complexity: C(t) ∈ [0, 1]

        Returns:
            (signal, dwell_counter)
            signal:         HysteresisSignal.STAY / REQUEST_DFN / REQUEST_DTLN
            dwell_counter:  current consecutive-frame count for the pending condition
        """
        if not (0.0 <= complexity <= 1.0):
            # Invalid complexity → safe fallback: stay with current model
            self._dwell_counter = 0
            self._pending_direction = None
            return HysteresisSignal.STAY, 0

        # Determine what direction we are pushing toward
        if self._current_model == ModelID.DTLN:
            # From DTLN, only switch UP (to DFN) if above T_high
            if complexity > self._t_high:
                new_direction = "up"
            else:
                new_direction = None  # in zone or below (can't go further down)

        elif self._current_model == ModelID.DEEP_FILTER_NET:
            # From DFN, only switch DOWN (to DTLN) if below T_low
            if complexity < self._t_low:
                new_direction = "down"
            else:
                new_direction = None  # in zone or above (can't go further up)

        else:
            # NONE or unknown → stay
            self._dwell_counter = 0
            self._pending_direction = None
            return HysteresisSignal.STAY, 0

        # Handle direction consistency
        if new_direction != self._pending_direction:
            # Direction changed (or cleared): reset dwell counter
            self._pending_direction = new_direction
            self._dwell_counter = 1 if new_direction is not None else 0
        elif new_direction is not None:
            # Same direction: increment dwell counter
            self._dwell_counter += 1
        else:
            # No direction: reset
            self._dwell_counter = 0

        # Check if dwell requirement is met
        if self._pending_direction == "up":
            required = self._dwell_enter
            if self._dwell_counter >= required:
                return HysteresisSignal.REQUEST_DFN, self._dwell_counter
        elif self._pending_direction == "down":
            required = self._dwell_exit
            if self._dwell_counter >= required:
                return HysteresisSignal.REQUEST_DTLN, self._dwell_counter

        return HysteresisSignal.STAY, self._dwell_counter


# ---------------------------------------------------------------------------
# Adaptive Threshold (future extension hook)
# ---------------------------------------------------------------------------

class AdaptiveThreshold:
    """
    Hook for future adaptive threshold computation.

    Currently returns fixed thresholds from config.
    Can be extended to adjust T_low and T_high based on long-term
    complexity statistics (e.g., shift thresholds if environment is
    consistently noisy).

    FUTURE:
        - Adjust T_high upward if environment is persistently complex
          (avoid constantly running expensive DFN)
        - Adjust T_low downward if environment is persistently quiet
          (avoid premature DFN activation)

    NOTE: Threshold adaptation must be validated carefully to avoid
    instability or degenerate lock-in behavior.
    """

    def __init__(self, config: AcousticConfig):
        self._base_low = config.threshold_low
        self._base_high = config.threshold_high
        self._adaptive_enabled = False  # disabled until validated

    @property
    def threshold_low(self) -> float:
        return self._base_low

    @property
    def threshold_high(self) -> float:
        return self._base_high

    def update(self, complexity: float) -> None:
        """Update adaptive thresholds (currently a no-op)."""
        # Future: update thresholds based on long-term complexity distribution
        pass
