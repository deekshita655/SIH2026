"""
metrics.py
==========
Metric collection and reporting for the SIH2026 integration pipeline.

===========================================================================
METRIC TAXONOMY — READ THIS FIRST
===========================================================================

Three distinct SNR concepts exist in this system.  They MUST NOT be
conflated:

    1. Dataset ground-truth SNR
       The target SNR used when generating the noisy mixture
       (e.g. "5 dB stationary").  Stored in dataset_metadata_1000.csv.
       This is the construction SNR — NOT a measurement of enhancement.

    2. NLMS-derived proxy SNR
       Derived at runtime from signal/noise power of the NLMS output.
       Labelled "NLMS-derived estimated SNR (NOT ground-truth)".
       Useful for monitoring adaptation quality.
       MUST NOT be used as the enhancement quality metric.

    3. Objective enhancement metrics (future)
       Compare clean reference vs. enhanced output using standard
       speech-quality measures: SNR, SI-SDR, STOI, PESQ.
       These require the clean reference audio to be loaded separately.
       They are NOT computed by default (library dependencies may not be
       installed).  Infrastructure is provided; values are populated when
       libraries are available.

Do NOT use output RMS decrease as evidence of enhancement quality.
Do NOT use AGC to artificially improve objective metrics.

===========================================================================
OBJECTIVE METRICS — LIBRARY AVAILABILITY
===========================================================================

    SNR, SI-SDR:  computed from numpy (always available)
    STOI:         requires pystoi   (pip install pystoi)
    PESQ:         requires pesq     (pip install pesq)

If a library is not installed the metric is reported as "not available",
not as a fabricated value.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .nlms import NLMSResult
from .pipeline import FrameDiagnostics, PipelineResult


# ---------------------------------------------------------------------------
# Basic audio quality
# ---------------------------------------------------------------------------

def compute_audio_metrics(wav: np.ndarray) -> dict:
    """Compute basic audio quality metrics for any waveform."""
    wav  = np.asarray(wav, dtype=np.float64)
    rms  = float(np.sqrt(np.mean(wav ** 2)))
    peak = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
    return {
        "rms":        rms,
        "rms_db":     float(20.0 * np.log10(rms + 1e-12)),
        "peak":       peak,
        "has_nan":    bool(np.any(np.isnan(wav))),
        "has_inf":    bool(np.any(np.isinf(wav))),
        "is_silence": rms < 1e-6,
        "length_samples": len(wav),
    }


# ---------------------------------------------------------------------------
# Objective enhancement metrics
# ---------------------------------------------------------------------------

def compute_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB) between reference and degraded signal.

    SNR = 10 * log10( sum(ref^2) / sum((ref - deg)^2) )

    Args:
        reference: clean reference waveform
        degraded:  noisy or processed waveform (same length as reference)

    Returns:
        SNR in dB.
    """
    reference = np.asarray(reference, dtype=np.float64)
    degraded  = np.asarray(degraded,  dtype=np.float64)
    n = min(len(reference), len(degraded))
    ref = reference[:n]
    deg = degraded[:n]
    sig_power  = float(np.mean(ref ** 2))
    noise_power= float(np.mean((ref - deg) ** 2))
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(sig_power / noise_power))


def compute_si_sdr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    SI-SDR = 10 * log10( ||alpha * s||^2 / ||alpha * s - s_hat||^2 )
    where alpha = <s_hat, s> / <s, s>

    Args:
        reference: clean reference waveform
        degraded:  estimated / enhanced waveform

    Returns:
        SI-SDR in dB.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    degraded  = np.asarray(degraded,  dtype=np.float64).ravel()
    n = min(len(reference), len(degraded))
    s = reference[:n] - reference[:n].mean()
    s_hat = degraded[:n] - degraded[:n].mean()
    alpha = float(np.dot(s_hat, s) / (np.dot(s, s) + 1e-12))
    s_target = alpha * s
    e_noise  = s_hat - s_target
    si_sdr = 10.0 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-12) + 1e-12
    )
    return float(si_sdr)


def compute_stoi(
    reference: np.ndarray,
    degraded:  np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    """
    Short-Time Objective Intelligibility (STOI).

    Requires pystoi (pip install pystoi).
    Returns None if pystoi is not installed — does NOT fabricate a value.

    Args:
        reference:   clean reference waveform
        degraded:    processed / noisy waveform
        sample_rate: audio sample rate in Hz

    Returns:
        STOI score in [0, 1], or None if pystoi is unavailable.
    """
    try:
        from pystoi import stoi
    except ImportError:
        return None
    n = min(len(reference), len(degraded))
    ref = np.asarray(reference[:n], dtype=np.float64)
    deg = np.asarray(degraded[:n],  dtype=np.float64)
    try:
        return float(stoi(ref, deg, sample_rate, extended=False))
    except Exception:
        return None


def compute_pesq(
    reference:   np.ndarray,
    degraded:    np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    """
    Perceptual Evaluation of Speech Quality (PESQ).

    Requires pesq (pip install pesq).
    Returns None if pesq is not installed — does NOT fabricate a value.

    Supported sample rates: 8000 Hz (narrowband) or 16000 Hz (wideband).

    Args:
        reference:   clean reference waveform
        degraded:    processed / noisy waveform
        sample_rate: 8000 or 16000

    Returns:
        PESQ MOS-LQO score, or None if pesq is unavailable or rate unsupported.
    """
    try:
        from pesq import pesq
    except ImportError:
        return None
    if sample_rate not in (8000, 16000):
        return None
    mode = "wb" if sample_rate == 16000 else "nb"
    n    = min(len(reference), len(degraded))
    ref  = np.asarray(reference[:n], dtype=np.float32)
    deg  = np.asarray(degraded[:n],  dtype=np.float32)
    try:
        return float(pesq(sample_rate, ref, deg, mode))
    except Exception:
        return None


def compute_objective_metrics(
    clean:       np.ndarray,
    noisy:       np.ndarray,
    enhanced:    np.ndarray,
    nlms_output: np.ndarray,
    sample_rate: int,
) -> dict:
    """
    Compute objective speech quality metrics comparing clean reference vs
    each pipeline stage.

    Comparisons:
        clean vs noisy input        — baseline (how bad is the problem)
        clean vs NLMS output        — P2 contribution
        clean vs enhanced output    — end-to-end contribution

    Available metrics:
        SNR       — always computed (numpy only)
        SI-SDR    — always computed (numpy only)
        STOI      — computed if pystoi is installed
        PESQ      — computed if pesq is installed (16 kHz only)

    IMPORTANT:
        These metrics require the CLEAN reference audio, which is
        separate from the NLMS synthetic reference channel.
        The clean audio comes from clean_speech/ in the dataset.

    Returns:
        Nested dict with metrics per stage.
    """
    def _stage(ref: np.ndarray, deg: np.ndarray) -> dict:
        return {
            "snr_db":   round(compute_snr(ref, deg), 3),
            "si_sdr_db": round(compute_si_sdr(ref, deg), 3),
            "stoi":     (
                round(v, 4)
                if (v := compute_stoi(ref, deg, sample_rate)) is not None
                else "not available (install pystoi)"
            ),
            "pesq_mos": (
                round(v, 4)
                if (v := compute_pesq(ref, deg, sample_rate)) is not None
                else "not available (install pesq; 16 kHz only)"
            ),
        }

    return {
        "note": (
            "Objective metrics compare clean reference vs each pipeline stage. "
            "Dataset ground-truth SNR is the construction SNR, not an enhancement metric. "
            "NLMS-derived proxy SNR is NOT the same as objective SNR."
        ),
        "clean_vs_noisy":   _stage(clean, noisy),
        "clean_vs_nlms":    _stage(clean, nlms_output),
        "clean_vs_enhanced": _stage(clean, enhanced),
    }


# ---------------------------------------------------------------------------
# NLMS metrics formatting
# ---------------------------------------------------------------------------

def format_nlms_metrics(result: NLMSResult) -> dict:
    """Format NLMS metrics for display/saving."""
    return {
        "filter_length":         result.filter_length,
        "step_size":             result.step_size,
        "block_size":            result.block_size,
        "num_blocks":            result.num_blocks,
        "signal_length_samples": result.signal_length,
        "audio_duration_sec":    round(result.audio_duration_sec, 3),
        "sample_rate":           result.sample_rate,
        # RMS levels
        "primary_rms":           round(result.primary_rms,          6),
        "reference_rms":         round(result.reference_rms,        6),
        "estimated_noise_rms":   round(result.estimated_noise_rms,  6),
        "cleaned_speech_rms":    round(result.cleaned_speech_rms,   6),
        # SNR estimate — clearly labelled
        "nlms_derived_snr_estimate_db": (
            round(result.nlms_snr_estimate_db, 2)
            if result.nlms_snr_estimate_db != float("inf")
            else "inf"
        ),
        "snr_label": (
            "NLMS-derived proxy SNR — NOT ground-truth, NOT an enhancement metric"
        ),
        # Performance
        "processing_time_sec": round(result.processing_time_sec, 3),
        "real_time_factor":    round(result.real_time_factor,    3),
        # Validity
        "has_nan": result.has_nan,
        "has_inf": result.has_inf,
    }


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

def format_system_metrics(result: PipelineResult) -> dict:
    """Format system-level metrics."""
    frames       = result.frame_diagnostics
    complexities = [f.complexity_score for f in frames]
    alphas       = [f.crossfade_alpha  for f in frames]
    router_states= [f.router_state     for f in frames]

    state_counts: dict = {}
    for s in router_states:
        state_counts[s] = state_counts.get(s, 0) + 1

    transitions = sum(1 for s in router_states if s == "TRANSITION")

    return {
        "total_frames":          result.num_frames,
        "audio_duration_sec":    round(result.audio_duration_sec,   3),
        "output_length_samples": result.output_length,
        "sample_rate":           result.sample_rate,
        "total_processing_sec":  round(result.total_processing_sec, 3),
        "end_to_end_rtf":        round(result.real_time_factor,     3),
        "fallback_frames":       result.fallback_frame_count,
        "complexity": {
            "mean": round(float(np.mean(complexities)), 4) if complexities else None,
            "min":  round(float(np.min(complexities)),  4) if complexities else None,
            "max":  round(float(np.max(complexities)),  4) if complexities else None,
            "std":  round(float(np.std(complexities)),  4) if complexities else None,
        },
        "router": {
            "state_counts":       state_counts,
            "transition_frames":  transitions,
            "max_crossfade_alpha": round(float(max(alphas)), 4) if alphas else 0.0,
        },
    }


def format_model_performance(result: PipelineResult) -> dict:
    """Format per-model performance statistics."""
    dur = result.audio_duration_sec
    return {
        "note": (
            "Timing reflects development-machine (desktop/laptop) execution. "
            "Do NOT use to estimate Raspberry Pi or Jetson Orin Nano performance."
        ),
        "dtln_slot": {
            "name":    result.dtln_metadata.name,
            "is_stub": result.dtln_metadata.is_stub,
            "is_trained": result.dtln_metadata.is_trained,
            "backend": result.dtln_metadata.backend,
            **result.dtln_stats.as_dict(dur),
        },
        "dfn_slot": {
            "name":    result.dfn_metadata.name,
            "is_stub": result.dfn_metadata.is_stub,
            "is_trained": result.dfn_metadata.is_trained,
            "backend": result.dfn_metadata.backend,
            **result.dfn_stats.as_dict(dur),
        },
    }


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------

def save_frame_diagnostics_csv(
    frame_diagnostics: List[FrameDiagnostics],
    output_path: str,
) -> None:
    """Save per-frame diagnostics to CSV."""
    import csv
    if not frame_diagnostics:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index", "timestamp_sec", "nlms_snr_db",
        "rms", "zcr", "spectral_flux", "spectral_entropy",
        "centroid_variation", "transient_score", "estimated_snr_db",
        "complexity_score", "router_state", "active_model",
        "crossfade_alpha", "is_transient", "dwell_counter",
        "model_fallback_used",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fd in frame_diagnostics:
            row = {
                "frame_index":        fd.frame_index,
                "timestamp_sec":      round(fd.timestamp_sec,        6),
                "nlms_snr_db":        round(fd.nlms_snr_db,          2) if fd.nlms_snr_db is not None else "",
                "rms":                round(fd.rms,                  6),
                "zcr":                round(fd.zcr,                  6),
                "spectral_flux":      round(fd.spectral_flux,        6),
                "spectral_entropy":   round(fd.spectral_entropy,     6),
                "centroid_variation": round(fd.centroid_variation,   6),
                "transient_score":    round(fd.transient_score,      6),
                "estimated_snr_db":   round(fd.estimated_snr_db,     4),
                "complexity_score":   round(fd.complexity_score,     6),
                "router_state":       fd.router_state,
                "active_model":       fd.active_model,
                "crossfade_alpha":    round(fd.crossfade_alpha,      4),
                "is_transient":       fd.is_transient,
                "dwell_counter":      fd.dwell_counter,
                "model_fallback_used": fd.model_fallback_used,
            }
            writer.writerow(row)


def save_metrics_json(
    nlms_metrics:   dict,
    system_metrics: dict,
    audio_metrics:  dict,
    output_path:    str,
    extra:          Optional[dict] = None,
) -> None:
    """Save all metrics to a JSON file."""
    report = {
        "sih2026_integration": {
            "version":  "0.2.0",
            "maturity": "Software development demonstrator",
            "disclaimer": (
                "16 kHz development dataset — 1000 noisy/clean pairs. "
                "32 kHz and 48 kHz are configuration/interface support only "
                "(no dedicated dataset). "
                "Raspberry Pi 5 / Jetson Orin Nano performance has NOT been validated. "
                "Model slots currently hold development stubs (no trained models)."
            ),
        },
        "nlms_metrics":   nlms_metrics,
        "system_metrics": system_metrics,
        "audio_quality":  audio_metrics,
    }
    if extra:
        report.update(extra)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
