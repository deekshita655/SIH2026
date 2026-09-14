"""
crossfade.py
============
Crossfade controller for model transitions in Person 5.

During a model switch (DTLN → DFN or vice versa), the router does NOT
abruptly cut from one model to the other. Instead, a crossfade blends
the outputs over N frames:

    S_t = (1 - alpha) * S_old + alpha * S_new

where alpha ramps from 0.0 to 1.0 over `transition_frames` frames.

The crossfade controller is INDEPENDENT of the actual model implementations.
It only tracks the alpha value. The actual blending of audio output is done
by Person 4 (output stage), which receives (alpha, active_model, target_model).

Design:
    - Linear ramp by default (simple, deterministic, C++-friendly)
    - Equal-power ramp option for perceptually smoother transitions
    - Configurable transition length (default: 5 frames = 50 ms at 10 ms hop)
    - Clean start (alpha = 0), clean end (alpha = 1.0)
    - After transition_frames, transition is complete

The router calls:
    crossfade.start_transition(from_model, to_model)  → begin ramp
    crossfade.step()                                   → advance one frame, get alpha
    crossfade.is_complete                              → True when alpha reached 1.0
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional

from .interfaces import ModelID


# ---------------------------------------------------------------------------
# Ramp type
# ---------------------------------------------------------------------------

class RampType(Enum):
    """Crossfade ramp shape."""
    LINEAR = auto()          # alpha = frame / total
    EQUAL_POWER = auto()     # alpha = sin(pi/2 * frame/total)^2 — perceptually flat


# ---------------------------------------------------------------------------
# Crossfade Controller
# ---------------------------------------------------------------------------

class CrossfadeController:
    """
    Controls the alpha blending ramp during model transitions.

    State:
        - NOT_TRANSITIONING: idle, alpha fixed at 0.0 or 1.0
        - TRANSITIONING:     ramping alpha from 0.0 to 1.0

    Usage:
        cf = CrossfadeController(transition_frames=5)

        # Start a transition
        cf.start_transition(from_model=ModelID.DTLN, to_model=ModelID.DEEP_FILTER_NET)

        # Each frame during transition:
        while not cf.is_complete:
            alpha = cf.step()
            # alpha goes 0.0 → ... → 1.0 over transition_frames

        # After completion:
        cf.finish()   # explicitly clear the transition state

    Args:
        transition_frames:  number of frames for the full ramp (default: 5)
        ramp_type:          shape of the ramp (default: LINEAR)
    """

    def __init__(
        self,
        transition_frames: int = 5,
        ramp_type: RampType = RampType.LINEAR,
    ):
        if transition_frames < 1:
            raise ValueError("transition_frames must be >= 1")
        self._total_frames = transition_frames
        self._ramp_type = ramp_type

        self._transitioning: bool = False
        self._frame: int = 0             # frames completed so far [0, total]
        self._from_model: Optional[ModelID] = None
        self._to_model: Optional[ModelID] = None

    def reset(self) -> None:
        """Reset to idle state."""
        self._transitioning = False
        self._frame = 0
        self._from_model = None
        self._to_model = None

    @property
    def is_transitioning(self) -> bool:
        return self._transitioning

    @property
    def is_complete(self) -> bool:
        """True when the ramp has finished (alpha reached 1.0)."""
        return self._transitioning and self._frame >= self._total_frames

    @property
    def alpha(self) -> float:
        """
        Current crossfade alpha ∈ [0.0, 1.0].

        0.0 = fully old model
        1.0 = fully new model
        """
        if not self._transitioning:
            return 0.0
        return self._compute_alpha(self._frame)

    @property
    def from_model(self) -> Optional[ModelID]:
        return self._from_model

    @property
    def to_model(self) -> Optional[ModelID]:
        return self._to_model

    @property
    def transition_frame(self) -> int:
        """Current frame number within the transition (0-indexed)."""
        return self._frame

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def start_transition(
        self,
        from_model: ModelID,
        to_model: ModelID,
    ) -> None:
        """
        Begin a crossfade from `from_model` to `to_model`.

        Args:
            from_model: model currently active (fades out)
            to_model:   target model (fades in)

        Raises:
            ValueError: if already transitioning or if from == to
        """
        if self._transitioning:
            raise RuntimeError(
                "CrossfadeController: start_transition called while "
                "already transitioning. Call finish() first."
            )
        if from_model == to_model:
            raise ValueError("from_model and to_model must differ")

        self._from_model = from_model
        self._to_model = to_model
        self._frame = 0
        self._transitioning = True

    def step(self) -> float:
        """
        Advance one frame in the transition and return the new alpha.

        Must only be called while is_transitioning is True.

        Returns:
            alpha ∈ [0.0, 1.0]

        Raises:
            RuntimeError: if called when not transitioning
        """
        if not self._transitioning:
            raise RuntimeError(
                "CrossfadeController.step() called when not transitioning"
            )
        if self._frame < self._total_frames:
            self._frame += 1
        alpha = self._compute_alpha(self._frame)
        return alpha

    def finish(self) -> None:
        """
        Explicitly complete the transition.

        Called by the router when it has confirmed the new model is active.
        Resets the crossfade controller to idle.
        """
        self._transitioning = False
        self._frame = self._total_frames  # mark complete
        # Do NOT clear _from_model/_to_model: useful for diagnostics

    def _compute_alpha(self, frame: int) -> float:
        """
        Compute alpha at `frame` within the transition.

        frame = 0     → alpha = 0.0 (fully old model)
        frame = total → alpha = 1.0 (fully new model)
        """
        if self._total_frames == 0:
            return 1.0
        t = min(float(frame) / float(self._total_frames), 1.0)
        if self._ramp_type == RampType.LINEAR:
            return float(t)
        elif self._ramp_type == RampType.EQUAL_POWER:
            # sin²(π/2 * t): starts at 0, ends at 1, smooth in between
            return float(math.sin(math.pi / 2.0 * t) ** 2)
        return float(t)  # fallback to linear


# ---------------------------------------------------------------------------
# Blend helper (for P4 / output stage)
# ---------------------------------------------------------------------------

def blend(
    output_old: float,
    output_new: float,
    alpha: float,
) -> float:
    """
    Linear crossfade blend of two scalar outputs.

        S_t = (1 - alpha) * S_old + alpha * S_new

    Used by Person 4 to blend model outputs during transition.
    P5 exports this as a utility function.

    Args:
        output_old: output from the departing model
        output_new: output from the incoming model
        alpha:      crossfade factor ∈ [0, 1]

    Returns:
        blended scalar output
    """
    alpha = max(0.0, min(1.0, alpha))
    return (1.0 - alpha) * output_old + alpha * output_new
