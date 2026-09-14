"""
router.py
=========
Model Router State Machine for Person 5.

The router owns complexity-based model selection, model availability and
crossfade state. It never performs model inference or DSP reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .interfaces import AcousticConfig, ModelID, RouterState, RouterDecision
from .hysteresis import HysteresisController, HysteresisSignal
from .crossfade import CrossfadeController


@dataclass
class ModelHealth:
    """Availability and health state for one model slot."""
    model_id: ModelID
    available: bool = True
    healthy: bool = True
    failure_count: int = 0

    @property
    def is_usable(self) -> bool:
        return self.available and self.healthy

    def report_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= 3:
            self.healthy = False

    def reset_health(self) -> None:
        self.failure_count = 0
        self.healthy = True


@dataclass
class RouterConfig:
    """Router-specific configuration, derived from AcousticConfig."""
    startup_frames: int = 10
    default_model: ModelID = ModelID.DTLN
    fallback_model: ModelID = ModelID.DTLN

    @classmethod
    def from_config(cls, cfg: AcousticConfig) -> "RouterConfig":
        return cls(
            startup_frames=cfg.startup_frames,
            default_model=ModelID.DTLN,
            fallback_model=ModelID.DTLN,
        )


class ModelRouter:
    """Explicit DTLN/DeepFilterNet router with availability-aware startup."""

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
        self._crossfade = CrossfadeController(transition_frames=config.crossfade_frames)
        self._state = RouterState.STARTUP
        self._active_model = self._router_cfg.default_model
        self._requested_model = self._router_cfg.default_model
        self._target_model: Optional[ModelID] = None
        self._frame_count = 0
        self._health: dict[ModelID, ModelHealth] = {
            ModelID.DTLN: ModelHealth(ModelID.DTLN, available=True),
            ModelID.DEEP_FILTER_NET: ModelHealth(ModelID.DEEP_FILTER_NET, available=False),
            ModelID.NONE: ModelHealth(ModelID.NONE, available=True),
        }

    def reset(self) -> None:
        self._hysteresis.reset(self._router_cfg.default_model)
        self._crossfade.reset()
        self._state = RouterState.STARTUP
        self._active_model = self._router_cfg.default_model
        self._requested_model = self._router_cfg.default_model
        self._target_model = None
        self._frame_count = 0

    def set_model_available(self, model: ModelID, available: bool) -> None:
        if model in self._health:
            self._health[model].available = available

    def report_model_failure(self, model: ModelID) -> None:
        if model in self._health:
            self._health[model].report_failure()
            if not self._health[model].is_usable and model == self._active_model:
                self._trigger_fallback()

    def reset_model_health(self, model: ModelID) -> None:
        if model in self._health:
            self._health[model].reset_health()

    def model_availability(self) -> dict[str, bool]:
        return {m.value: h.is_usable for m, h in self._health.items()}

    def process(self, complexity: float, frame_index: int) -> RouterDecision:
        self._frame_count += 1
        import math
        if math.isnan(complexity) or math.isinf(complexity):
            return self._make_decision(frame_index, complexity, override_state=self._state)

        if self._state == RouterState.STARTUP:
            self._handle_startup()
        elif self._state in (RouterState.DTLN, RouterState.DFN):
            self._handle_steady_state(complexity)
        elif self._state == RouterState.TRANSITION:
            self._handle_transition()
        elif self._state == RouterState.FALLBACK:
            self._handle_fallback(complexity)
        return self._make_decision(frame_index, complexity)

    def _handle_startup(self) -> None:
        preferred = self._router_cfg.default_model

        # Never emit an unavailable active model, even during the startup
        # hold period.  If the preferred slot is unavailable, select another
        # usable model immediately; otherwise retain the normal startup
        # dwell before committing to the preferred model.
        if not self._health[preferred].is_usable:
            if self._health[ModelID.DEEP_FILTER_NET].is_usable:
                selected = ModelID.DEEP_FILTER_NET
            elif self._health[ModelID.NONE].is_usable:
                selected = ModelID.NONE
            else:
                self._state = RouterState.FALLBACK
                self._active_model = ModelID.NONE
                self._requested_model = ModelID.NONE
                return
            self._active_model = selected
            self._requested_model = selected
            self._state = (
                RouterState.DFN
                if selected == ModelID.DEEP_FILTER_NET
                else RouterState.FALLBACK
            )
            return

        if self._frame_count < self._router_cfg.startup_frames:
            return

        self._active_model = preferred
        self._requested_model = preferred
        self._state = RouterState.DTLN

    def _handle_steady_state(self, complexity: float) -> None:
        signal, _ = self._hysteresis.update(complexity)
        if signal == HysteresisSignal.REQUEST_DFN:
            target = ModelID.DEEP_FILTER_NET
            if self._health[target].is_usable:
                self._start_transition(self._active_model, target)
            else:
                self._hysteresis.reset(self._active_model)
        elif signal == HysteresisSignal.REQUEST_DTLN:
            target = ModelID.DTLN
            if self._health[target].is_usable:
                self._start_transition(self._active_model, target)
            else:
                self._trigger_fallback()

    def _handle_transition(self) -> None:
        if self._crossfade.is_complete:
            new_model = self._target_model
            if new_model is None or not self._health[new_model].is_usable:
                self._trigger_fallback()
                return
            self._active_model = new_model
            self._requested_model = new_model
            self._target_model = None
            self._state = RouterState.DTLN if new_model == ModelID.DTLN else RouterState.DFN
            self._hysteresis.notify_switch_complete(new_model)
            self._crossfade.finish()
        else:
            self._crossfade.step()

    def _handle_fallback(self, complexity: float) -> None:
        if self._health[ModelID.DTLN].is_usable:
            self._active_model = ModelID.DTLN
            self._requested_model = ModelID.DTLN
            self._state = RouterState.DTLN
            self._hysteresis.reset(ModelID.DTLN)
        elif self._health[ModelID.DEEP_FILTER_NET].is_usable:
            self._active_model = ModelID.DEEP_FILTER_NET
            self._requested_model = ModelID.DEEP_FILTER_NET
            self._state = RouterState.DFN
            self._hysteresis.reset(ModelID.DEEP_FILTER_NET)

    def _start_transition(self, from_model: ModelID, to_model: ModelID) -> None:
        if not self._health[to_model].is_usable:
            return
        self._target_model = to_model
        self._requested_model = to_model
        self._state = RouterState.TRANSITION
        try:
            self._crossfade.start_transition(from_model, to_model)
        except RuntimeError:
            pass

    def _trigger_fallback(self) -> None:
        fallback = self._router_cfg.fallback_model
        if not self._health[fallback].is_usable:
            fallback = ModelID.DEEP_FILTER_NET if self._health[ModelID.DEEP_FILTER_NET].is_usable else ModelID.NONE
        self._active_model = fallback
        self._requested_model = fallback
        self._state = RouterState.FALLBACK
        self._crossfade.reset()
        self._target_model = None

    def _make_decision(self, frame_index: int, complexity: float, override_state: Optional[RouterState] = None) -> RouterDecision:
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
