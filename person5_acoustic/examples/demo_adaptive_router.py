"""
demo_adaptive_router.py
=======================
Demonstrates the adaptive statistics and model router responding to a
changing acoustic environment.

Scenario:
    Phase 1: Quiet environment  -> DTLN active
    Phase 2: Noisy environment  -> complexity rises -> transitions to DFN
    Phase 3: Quiet again        -> complexity drops -> transitions back to DTLN

Uses synthetic signals only — no actual DTLN or DeepFilterNet models.
"""

import sys
import os
import numpy as np
import dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    AcousticPipeline,
    ModelID,
    RouterState,
)


def make_frame(idx: int, amp: float, sr: int = 16000, n: int = 320) -> AcousticFrame:
    rng = np.random.default_rng(idx)
    waveform = (rng.standard_normal(n) * amp).astype(np.float32)
    fft_size = 512
    magnitude = np.abs(np.fft.rfft(waveform, n=fft_size)).astype(np.float32)
    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sr).astype(np.float32)
    return AcousticFrame(
        frame_index=idx,
        sample_rate=sr,
        waveform=waveform,
        magnitude=magnitude,
        freq_bins=freq_bins,
        frame_length=n,
        hop_length=n // 2,
    )


def format_bar(value: float, width: int = 30) -> str:
    """Simple ASCII bar chart for complexity visualization."""
    filled = int(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.3f}"


def main():
    print("\n" + "="*70)
    print("  Person 5 — Adaptive Router Demo")
    print("  Scenario: Quiet -> Noisy -> Quiet environment transitions")
    print("="*70)

    # Short dwell/crossfade for demo purposes
    config = dataclasses.replace(
        AcousticConfig.development_16khz(),
        dwell_enter=5,
        dwell_exit=5,
        crossfade_frames=5,
        threshold_low=0.35,
        threshold_high=0.60,
    )

    pipeline = AcousticPipeline(config)
    pipeline.set_model_available(ModelID.DEEP_FILTER_NET, True)

    print(f"\n  Config: T_low={config.threshold_low}, T_high={config.threshold_high}")
    print(f"  dwell_enter={config.dwell_enter}, dwell_exit={config.dwell_exit}")
    print(f"  crossfade_frames={config.crossfade_frames}")
    print(f"  Hop length: {config.hop_length} samples = {1000*config.hop_length/config.sample_rate:.1f} ms/frame\n")

    print(f"  {'Frame':>5}  {'Phase':<12}  {'Complexity':<35}  {'State':<14}  {'Model':<14}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*35}  {'-'*14}  {'-'*14}")

    history = []

    def run_phase(start, count, amp, label):
        for i in range(start, start + count):
            frame = make_frame(i, amp=amp)
            diag = pipeline.process(frame)
            rd = diag.router_decision
            bar = format_bar(diag.complexity_score)
            state_str = rd.router_state.name
            model_str = rd.active_model.value
            if rd.router_state == RouterState.TRANSITION:
                model_str += f"->{rd.target_model.value if rd.target_model else '?'}"
                model_str = model_str[:18]
            print(f"  {i:>5}  {label:<12}  {bar}  {state_str:<14}  {model_str:<18}")
            history.append({
                "frame": i, "phase": label,
                "complexity": diag.complexity_score,
                "state": state_str,
                "model": rd.active_model.value,
            })
        return start + count

    # Phase 1: Quiet (40 frames)
    idx = run_phase(0, 40, amp=0.01, label="QUIET")

    # Phase 2: Noisy (60 frames)
    idx = run_phase(idx, 60, amp=2.0, label="NOISY")

    # Phase 3: Quiet again (60 frames)
    idx = run_phase(idx, 60, amp=0.005, label="QUIET again")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    dtln_frames = sum(1 for h in history if h["model"] == "DTLN")
    dfn_frames = sum(1 for h in history if h["model"] == "DeepFilterNet")
    print(f"  Total frames:       {len(history)}")
    print(f"  Frames with DTLN:   {dtln_frames}")
    print(f"  Frames with DFN:    {dfn_frames}")

    transitions = [
        h for i, h in enumerate(history[1:], 1)
        if history[i]["model"] != history[i-1]["model"]
    ]
    print(f"  Model switches:     {len(transitions)}")
    for t in transitions:
        print(f"    Frame {t['frame']:4d}: -> {t['model']}")

    print("\n  NOTE: Complexity thresholds are ILLUSTRATIVE ONLY.")
    print("  Final T_low/T_high must be calibrated on a dev set.")
    print("  This demo uses synthetic noise — not real speech/noise.\n")


if __name__ == "__main__":
    main()
