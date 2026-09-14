"""Lightweight reference-signal quality estimator for NLMS control."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReferenceQualityConfig:
    """Thresholds for mapping reference quality to NLMS adaptation strength."""
    enabled: bool = True
    good_threshold: float = 0.65
    moderate_threshold: float = 0.35
    moderate_mu_scale: float = 0.30
    epsilon: float = 1e-10


@dataclass(frozen=True)
class ReferenceQualityResult:
    score: float
    state: str
    mu_scale: float
    correlation: float
    coherence: float
    energy_ratio: float


def _correlation(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    x = x[:n].astype(np.float64, copy=False) - np.mean(x[:n])
    y = y[:n].astype(np.float64, copy=False) - np.mean(y[:n])
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y)) + eps
    return float(np.clip(abs(np.dot(x, y)) / denom, 0.0, 1.0))


def _coherence(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Estimate mean magnitude-squared coherence using short Welch segments."""
    n = min(x.size, y.size)
    if n < 16:
        return 0.0
    x = x[:n].astype(np.float64, copy=False)
    y = y[:n].astype(np.float64, copy=False)
    seg_len = min(256, n)
    if seg_len < 16:
        return 0.0
    step = max(seg_len // 2, 1)
    window = np.hanning(seg_len)
    pxx = pyy = pxy = None
    count = 0
    for start in range(0, n - seg_len + 1, step):
        xs = (x[start:start + seg_len] - np.mean(x[start:start + seg_len])) * window
        ys = (y[start:start + seg_len] - np.mean(y[start:start + seg_len])) * window
        X, Y = np.fft.rfft(xs), np.fft.rfft(ys)
        if pxx is None:
            pxx, pyy, pxy = abs(X) ** 2, abs(Y) ** 2, X * np.conj(Y)
        else:
            pxx += abs(X) ** 2
            pyy += abs(Y) ** 2
            pxy += X * np.conj(Y)
        count += 1
    if count == 0:
        return 0.0
    pxx /= count
    pyy /= count
    pxy /= count
    coherence = abs(pxy) ** 2 / (pxx * pyy + eps)
    valid = (pxx > eps) & (pyy > eps)
    return float(np.clip(np.mean(coherence[valid]) if np.any(valid) else 0.0, 0.0, 1.0))


def estimate_reference_quality(primary: np.ndarray, reference: np.ndarray,
                               config: ReferenceQualityConfig | None = None) -> ReferenceQualityResult:
    """Return a bounded quality score; this is a controller signal, not ground-truth SNR."""
    cfg = config or ReferenceQualityConfig()
    if not cfg.enabled:
        return ReferenceQualityResult(1.0, "GOOD", 1.0, 1.0, 1.0, 1.0)
    p = np.asarray(primary, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    n = min(p.size, r.size)
    if n == 0:
        return ReferenceQualityResult(0.0, "POOR", 0.0, 0.0, 0.0, 0.0)
    p, r = p[:n], r[:n]
    corr = _correlation(p, r, cfg.epsilon)
    coh = _coherence(p, r, cfg.epsilon)
    ep, er = float(np.mean(p * p)), float(np.mean(r * r))
    ratio = float(np.clip(min(ep, er) / (max(ep, er) + cfg.epsilon), 0.0, 1.0)) if max(ep, er) > cfg.epsilon else 0.0
    score = float(np.clip(0.5 * corr + 0.3 * coh + 0.2 * ratio, 0.0, 1.0))
    if score >= cfg.good_threshold:
        return ReferenceQualityResult(score, "GOOD", 1.0, corr, coh, ratio)
    if score >= cfg.moderate_threshold:
        return ReferenceQualityResult(score, "MODERATE", cfg.moderate_mu_scale, corr, coh, ratio)
    return ReferenceQualityResult(score, "POOR", 0.0, corr, coh, ratio)
