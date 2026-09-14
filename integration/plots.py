"""
plots.py
========
Diagnostic plotting for the SIH2026 end-to-end pipeline.

Generates all required plots:
  1. Original noisy waveform
  2. NLMS cleaned waveform
  3. Final enhanced waveform
  4. Complexity C(t) over time
  5. Router state over time
  6. Crossfade alpha
  7. Input spectrogram (noisy)
  8. NLMS-cleaned spectrogram
  9. Output spectrogram (enhanced)

Uses matplotlib (non-interactive backend by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def _get_mpl():
    """Import matplotlib with non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive — safe on headless systems
    import matplotlib.pyplot as plt
    return matplotlib, plt


def plot_waveforms(
    primary: np.ndarray,
    nlms_cleaned: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str,
    show: bool = False,
) -> None:
    """Plot primary / NLMS-cleaned / enhanced waveforms."""
    mpl, plt = _get_mpl()
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t = np.arange(len(primary)) / sample_rate

    axes[0].plot(t, primary, color="#2196F3", linewidth=0.5, alpha=0.8)
    axes[0].set_title("Primary Microphone Signal (Noisy Speech)", fontsize=11)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, nlms_cleaned[:len(t)], color="#FF9800", linewidth=0.5, alpha=0.8)
    axes[1].set_title(
        "NLMS Cleaned Waveform e[n]  "
        "(Python dev implementation — P2 MATLAB NLMS reference)",
        fontsize=11,
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, enhanced[:len(t)], color="#4CAF50", linewidth=0.5, alpha=0.8)
    axes[2].set_title(
        "Final Enhanced Output  "
        "(P2 -> P3 STFT -> P5 Router -> P4 stub -> P3 ISTFT)",
        fontsize=11,
    )
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("SIH2026 — End-to-End Speech Enhancement (16 kHz dev)", fontsize=13, y=1.01)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_complexity_and_router(
    timestamps: np.ndarray,
    complexity: np.ndarray,
    router_states: List[str],
    alphas: np.ndarray,
    output_path: str,
    threshold_low: float = 0.35,
    threshold_high: float = 0.65,
    show: bool = False,
) -> None:
    """Plot complexity C(t), router state, and crossfade alpha."""
    mpl, plt = _get_mpl()

    # Map router states to numeric values for plotting
    state_map = {"STARTUP": 0, "DTLN": 1, "TRANSITION": 2, "DFN": 3, "FALLBACK": -1}
    state_numeric = np.array([state_map.get(s, 0) for s in router_states])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Plot 1: Complexity
    axes[0].plot(timestamps, complexity, color="#673AB7", linewidth=1.2, label="C(t)")
    axes[0].axhline(threshold_high, color="red", linestyle="--", linewidth=0.8,
                    label=f"T_high={threshold_high}")
    axes[0].axhline(threshold_low, color="orange", linestyle="--", linewidth=0.8,
                    label=f"T_low={threshold_low}")
    axes[0].fill_between(timestamps, threshold_low, threshold_high,
                         alpha=0.1, color="yellow", label="Hysteresis zone")
    axes[0].set_ylabel("Complexity C(t)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Acoustic Complexity Score C(t)  [P5 Acoustic Intelligence]", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Router state
    state_colors = {0: "#9E9E9E", 1: "#2196F3", 2: "#FF9800", 3: "#F44336", -1: "#795548"}
    for state_val, label in [(0, "STARTUP"), (1, "DTLN"), (2, "TRANSITION"),
                              (3, "DFN"), (-1, "FALLBACK")]:
        mask = state_numeric == state_val
        if np.any(mask):
            axes[1].scatter(timestamps[mask], np.full(mask.sum(), state_val),
                            c=state_colors[state_val], s=4, label=label, alpha=0.8)
    axes[1].set_yticks([-1, 0, 1, 2, 3])
    axes[1].set_yticklabels(["FALLBACK", "STARTUP", "DTLN", "TRANSITION", "DFN"])
    axes[1].set_ylabel("Router State")
    axes[1].legend(fontsize=7, loc="upper right", ncol=3)
    axes[1].set_title("Router State  [P5 Adaptive Model Router]", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Crossfade alpha
    axes[2].fill_between(timestamps, alphas, alpha=0.4, color="#FF9800")
    axes[2].plot(timestamps, alphas, color="#E65100", linewidth=1.0)
    axes[2].set_ylabel("Crossfade α")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Crossfade Alpha (0=DTLN, 1=DFN)  [P5 CrossfadeController]", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "SIH2026 — P5 Routing Diagnostics (16 kHz development dataset)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_spectrograms(
    primary: np.ndarray,
    nlms_cleaned: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str,
    show: bool = False,
) -> None:
    """Plot spectrograms of primary / NLMS-cleaned / enhanced signals.

    NFFT and noverlap are derived from sample_rate (20 ms window / 10 ms hop)
    so the function works correctly at 16, 32, and 48 kHz.
    """
    mpl, plt = _get_mpl()
    import warnings

    # Derive window sizes from sample_rate — no hard-coded constants
    nfft    = round(sample_rate * 20 / 1000)   # 20 ms window
    noverlap= round(sample_rate * 10 / 1000)   # 10 ms hop  → 50% overlap

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    common_kw = dict(
        Fs      = sample_rate,
        NFFT    = nfft,
        noverlap= noverlap,
        cmap    = "inferno",
        scale   = "dB",
    )

    for ax, wav, title in [
        (axes[0], primary,
         "Noisy Speech Spectrogram (Primary Input)"),
        (axes[1], nlms_cleaned[:len(primary)],
         "NLMS Cleaned Spectrogram (P2 NLMS Output)"),
        (axes[2], enhanced[:len(primary)],
         "Enhanced Output Spectrogram (P4 stub → P3 ISTFT)"),
    ]:
        try:
            # Suppress log10(0) RuntimeWarning — silent/zero frames produce
            # -inf dB bins which matplotlib logs a divide-by-zero for.
            # This is a display artefact only; the audio data is unaffected.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spec, freqs, times, im = ax.specgram(wav, **common_kw)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Freq (Hz)")
            fig.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.01)
        except Exception as e:
            ax.text(0.5, 0.5, f"Spectrogram error: {e}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)

    axes[2].set_xlabel("Time (s)")
    fig.suptitle(
        f"SIH2026 — Spectrograms Comparison ({sample_rate // 1000} kHz development)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_features(
    timestamps: np.ndarray,
    rms_vals: np.ndarray,
    zcr_vals: np.ndarray,
    flux_vals: np.ndarray,
    entropy_vals: np.ndarray,
    snr_vals: np.ndarray,
    output_path: str,
    show: bool = False,
) -> None:
    """Plot per-frame acoustic features extracted by P5."""
    mpl, plt = _get_mpl()

    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)

    for ax, data, label, color in [
        (axes[0], rms_vals, "RMS Energy", "#2196F3"),
        (axes[1], zcr_vals, "Zero Crossing Rate", "#4CAF50"),
        (axes[2], flux_vals, "Spectral Flux", "#FF9800"),
        (axes[3], entropy_vals, "Spectral Entropy", "#9C27B0"),
        (axes[4], snr_vals, "Estimated SNR (dB) — P5 proxy", "#F44336"),
    ]:
        ax.plot(timestamps, data, color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[4].set_xlabel("Time (s)")
    fig.suptitle(
        "SIH2026 — P5 Acoustic Features per Frame (16 kHz development)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
