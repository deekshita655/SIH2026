"""
adaptive_stats.py
=================
Lightweight online adaptive statistics for Person 5.

Implements:
    - RunningMedian:        finite-window running median using sorted insertion
    - RunningMAD:           Median Absolute Deviation, robust spread estimate
    - OutlierDetector:      |x - median| > k * MAD rule
    - ThresholdedEWMA:      EWMA that ignores outliers to protect baseline

Design goals:
    - Pure Python + NumPy only (no SciPy dependency)
    - Suitable for later C++ translation (simple state, explicit arithmetic)
    - Configurable window sizes and timescales
    - Robust to impulsive noise (outliers do not corrupt the baseline)

Two timescale design:
    FAST statistics:  react to immediate acoustic changes (smaller window / larger alpha)
    SLOW statistics:  track long-term environmental baseline (larger window / smaller alpha)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Running Median
# ---------------------------------------------------------------------------

class RunningMedian:
    """
    Finite-window running median.

    Maintains a sliding window of the last `window` values and returns
    the median of that window.

    Implementation:
        - Use a deque for O(1) append/pop of the sliding window.
        - For the median, sort the window contents (O(w log w)).
        - Window sizes are small (default 31 frames), so sorting is cheap.
        - Suitable for C++ translation: replace deque with a circular buffer.

    Args:
        window: number of frames to include (should be odd for clean median)
    """

    def __init__(self, window: int = 31):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def count(self) -> int:
        """Number of samples currently in the window."""
        return len(self._buf)

    @property
    def window(self) -> int:
        return self._window

    def update(self, x: float) -> float:
        """
        Insert new sample and return the current running median.

        Args:
            x: new scalar observation

        Returns:
            current median of the window (or x itself if only 1 sample)
        """
        self._buf.append(float(x))
        return self.value

    @property
    def value(self) -> float:
        """Current running median without inserting a new sample."""
        if len(self._buf) == 0:
            return 0.0
        # np.median on a small list is accurate and cheap
        return float(np.median(list(self._buf)))

    @property
    def is_warm(self) -> bool:
        """True when window is fully populated."""
        return len(self._buf) >= self._window


# ---------------------------------------------------------------------------
# Running MAD (Median Absolute Deviation)
# ---------------------------------------------------------------------------

class RunningMAD:
    """
    Median Absolute Deviation over a finite window.

        MAD_t = median( |x_i - median(x)| )   for x_i in the current window

    MAD is a robust measure of dispersion. Unlike standard deviation,
    it is resistant to outliers / impulsive noise.

    This class operates on the same window as the supplied RunningMedian,
    OR maintains its own independent buffer.
    """

    def __init__(self, window: int = 31):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def count(self) -> int:
        return len(self._buf)

    def update(self, x: float) -> tuple[float, float]:
        """
        Insert new sample and return (median, MAD).

        Returns:
            (median, MAD)  both as floats
        """
        self._buf.append(float(x))
        return self.values

    @property
    def values(self) -> tuple[float, float]:
        """Return (median, MAD) for the current window without inserting."""
        data = list(self._buf)
        if not data:
            return 0.0, 0.0
        arr = np.array(data)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med, mad

    @property
    def median(self) -> float:
        return self.values[0]

    @property
    def mad(self) -> float:
        return self.values[1]

    @property
    def is_warm(self) -> bool:
        return len(self._buf) >= self._window


# ---------------------------------------------------------------------------
# Outlier Detector
# ---------------------------------------------------------------------------

class OutlierDetector:
    """
    Robust outlier detection using the Median/MAD criterion.

        |x - median| > k * MAD  →  outlier

    This is analogous to the "3-sigma" rule but using robust statistics.
    A single impulsive event cannot redefine the baseline because it
    affects neither the median nor the MAD significantly.

    Args:
        window:  window for running MAD
        k:       outlier threshold multiplier (default 3.0)
        epsilon: minimum MAD to prevent division by near-zero spread
    """

    def __init__(
        self,
        window: int = 31,
        k: float = 3.0,
        epsilon: float = 1e-8,
    ):
        self._mad_tracker = RunningMAD(window=window)
        self._k = k
        self._epsilon = epsilon

    def reset(self) -> None:
        self._mad_tracker.reset()

    def update(self, x: float) -> tuple[bool, float, float]:
        """
        Insert sample and classify it.

        Args:
            x: new observation

        Returns:
            (is_outlier, median, mad)
        """
        self._mad_tracker.update(x)
        median, mad = self._mad_tracker.values
        is_outlier = abs(x - median) > self._k * max(mad, self._epsilon)
        return is_outlier, median, mad

    def is_outlier(self, x: float) -> bool:
        """Check without updating. Useful for peek-only inspection."""
        median, mad = self._mad_tracker.values
        return abs(x - median) > self._k * max(mad, self._epsilon)

    @property
    def is_warm(self) -> bool:
        return self._mad_tracker.is_warm


# ---------------------------------------------------------------------------
# EWMA  (Exponentially Weighted Moving Average)
# ---------------------------------------------------------------------------

class EWMA:
    """
    Standard Exponentially Weighted Moving Average.

        E_t = (1 - alpha) * E_(t-1) + alpha * x_t

    Suitable for C++ translation: only one float of state per instance.

    Args:
        alpha: smoothing factor in (0, 1]. Larger = faster adaptation.
        initial: optional initial value; if None, first sample initialises it.
    """

    def __init__(self, alpha: float, initial: Optional[float] = None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._value: Optional[float] = initial

    def reset(self, value: Optional[float] = None) -> None:
        self._value = value

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def is_initialized(self) -> bool:
        return self._value is not None

    def update(self, x: float) -> float:
        """
        Update EWMA with new sample.

        Returns:
            updated EWMA value
        """
        if self._value is None:
            self._value = float(x)
        else:
            self._value = (1.0 - self._alpha) * self._value + self._alpha * float(x)
        return self._value


# ---------------------------------------------------------------------------
# Thresholded EWMA  (outlier-gated baseline)
# ---------------------------------------------------------------------------

class ThresholdedEWMA:
    """
    EWMA that only accepts samples that pass an outlier gate.

    Purpose:
        Maintain a slowly-adapting environmental baseline that is NOT
        corrupted by transient spikes or impulsive noise events.

    Algorithm:
        1. Maintain a RunningMAD for robust local statistics.
        2. For each new sample x_t:
           a. Classify as outlier using MAD rule.
           b. If NOT outlier → update both EWMA and MAD window.
           c. If IS outlier  → update MAD window but NOT EWMA baseline.

    This means:
        - The median/MAD window sees all data (so it stays accurate).
        - The EWMA baseline only absorbs non-outlier data (so it stays clean).

    Two instances should be created per feature:
        FAST:  smaller window, larger alpha → tracks recent changes
        SLOW:  larger window, smaller alpha → tracks long-term baseline

    Args:
        alpha:   EWMA smoothing
        window:  RunningMAD window for outlier classification
        k:       outlier threshold multiplier
        epsilon: MAD floor
    """

    def __init__(
        self,
        alpha: float,
        window: int = 31,
        k: float = 3.0,
        epsilon: float = 1e-8,
    ):
        self._ewma = EWMA(alpha=alpha)
        self._outlier_det = OutlierDetector(window=window, k=k, epsilon=epsilon)
        self._alpha = alpha
        self._accepted_count: int = 0
        self._rejected_count: int = 0

    def reset(self) -> None:
        self._ewma.reset()
        self._outlier_det.reset()
        self._accepted_count = 0
        self._rejected_count = 0

    def update(self, x: float) -> tuple[float, bool]:
        """
        Update with new sample.

        Args:
            x: new observation

        Returns:
            (ewma_value, was_outlier)
        """
        is_outlier, _, _ = self._outlier_det.update(x)

        if not is_outlier:
            self._ewma.update(x)
            self._accepted_count += 1
        else:
            # Do NOT update EWMA with outlier value.
            # If EWMA is not yet initialised, initialise conservatively.
            if not self._ewma.is_initialized:
                self._ewma.update(x)  # first sample always accepted
            self._rejected_count += 1

        value = self._ewma.value if self._ewma.value is not None else x
        return float(value), is_outlier

    @property
    def value(self) -> Optional[float]:
        return self._ewma.value

    @property
    def is_initialized(self) -> bool:
        return self._ewma.is_initialized

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count


# ---------------------------------------------------------------------------
# Per-feature Adaptive Statistics Bundle
# ---------------------------------------------------------------------------

class FeatureAdaptiveStats:
    """
    Two-timescale adaptive statistics for a single scalar feature.

    Maintains:
        - running MAD (shared window for both timescales)
        - FAST ThresholdedEWMA (smaller alpha, reacts quickly)
        - SLOW ThresholdedEWMA (larger alpha, long-term baseline)

    Args:
        fast_alpha: EWMA alpha for the fast tracker
        slow_alpha: EWMA alpha for the slow tracker
        window:     window length for running MAD / outlier detection
        k:          outlier threshold (MAD multiplier)
        epsilon:    MAD floor
    """

    def __init__(
        self,
        fast_alpha: float = 0.10,
        slow_alpha: float = 0.01,
        window: int = 31,
        k: float = 3.0,
        epsilon: float = 1e-8,
    ):
        self._mad_tracker = RunningMAD(window=window)
        self._fast_ewma = EWMA(alpha=fast_alpha)
        self._slow_ewma = EWMA(alpha=slow_alpha)
        self._k = k
        self._epsilon = epsilon

    def reset(self) -> None:
        self._mad_tracker.reset()
        self._fast_ewma.reset()
        self._slow_ewma.reset()

    def update(self, x: float) -> "AdaptiveStatsSnapshot":
        """
        Update all trackers with a new observation.

        Returns:
            AdaptiveStatsSnapshot with all current statistics.
        """
        # Update MAD tracker (always, including outliers, so median stays accurate)
        median, mad = self._mad_tracker.update(x)

        # Outlier check
        is_outlier = abs(x - median) > self._k * max(mad, self._epsilon)

        # Update fast EWMA always (fast adaptation is the point)
        fast_val = self._fast_ewma.update(x)

        # Update slow EWMA only with non-outliers
        if not is_outlier:
            slow_val = self._slow_ewma.update(x)
        else:
            if not self._slow_ewma.is_initialized:
                slow_val = self._slow_ewma.update(x)
            else:
                slow_val = self._slow_ewma.value  # type: ignore[assignment]

        return AdaptiveStatsSnapshot(
            median=median,
            mad=mad,
            fast_ewma=float(fast_val) if fast_val is not None else float(x),
            slow_ewma=float(slow_val) if slow_val is not None else float(x),
            is_outlier=is_outlier,
        )


from dataclasses import dataclass


@dataclass
class AdaptiveStatsSnapshot:
    """Snapshot of adaptive statistics at a single time step."""
    median: float
    mad: float
    fast_ewma: float
    slow_ewma: float
    is_outlier: bool
