"""
demo_transition.py
==================
Demonstrates the crossfade controller during a DTLN -> DFN transition.

Shows:
    1. Router in DTLN steady state
    2. High-complexity frames trigger dwell counter
    3. Transition begins (DTLN -> DFN)
    4. Crossfade alpha ramps from 0 -> 1 over N frames
    5. DFN becomes active

Also shows the reverse (DFN -> DTLN) triggered by low complexity.

Uses synthetic signals only — no actual DTLN or DeepFilterNet models.
"""

import sys
import os
import dataclasses
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    AcousticPipeline,
    ModelID,
    RouterState,
)
from person5_acoustic.crossfade import CrossfadeController, RampType, blend


def make_frame(idx: int, amp: float, sr: int = 16000, n: int = 320) -> AcousticFrame:
    rng = np.random.default_rng(idx * 997 + int(amp * 1000))
    waveform = (rng.standard_normal(n) * amp).astype(np.float32)
    fft_size = 512
    magnitude = np.abs(np.fft.rfft(waveform, n=fft_size)).astype(np.float32)
    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sr).astype(np.float32)
    return AcousticFrame(
        frame_index=idx, sample_rate=sr,
        waveform=waveform, magnitude=magnitude, freq_bins=freq_bins,
        frame_length=n, hop_length=n // 2,
    )


def print_frame_row(frame_idx, phase, complexity, state, model, alpha, dwell):
    alpha_bar = "." * 10
    if state == "TRANSITION":
        filled = int(alpha * 10)
        alpha_bar = "#" * filled + "." * (10 - filled)

    print(f"  {frame_idx:>4}  {phase:<10}  C={complexity:.3f}  "
          f"{state:<12}  {model:<16}  alpha=[{alpha_bar}]{alpha:.2f}  "
          f"dwell={dwell}")


def demo_crossfade_standalone():
    print("\n" + "="*70)
    print("  Part 1: Standalone CrossfadeController demonstration")
    print("="*70)
    print("  Ramps alpha from 0.0 -> 1.0 over 8 frames (LINEAR and EQUAL_POWER)")
    print()

    for ramp_name, ramp_type in [
        ("LINEAR", RampType.LINEAR),
        ("EQUAL_POWER", RampType.EQUAL_POWER),
    ]:
        cf = CrossfadeController(transition_frames=8, ramp_type=ramp_type)
        cf.start_transition(ModelID.DTLN, ModelID.DEEP_FILTER_NET)

        print(f"  {ramp_name} ramp:")
        print(f"  {'Frame':>5}  {'Alpha':>8}  {'Blend Example (DTLN=0.0, DFN=1.0)':>12}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*40}")

        alphas = []
        while not cf.is_complete:
            alpha = cf.step()
            alphas.append(alpha)
            blended = blend(0.0, 1.0, alpha)  # example: DTLN output=0.0, DFN output=1.0
            bar_w = int(alpha * 30)
            bar = "#" * bar_w + "." * (30 - bar_w)
            print(f"  {cf.transition_frame:>5}  {alpha:>8.4f}  [{bar}]")

        cf.finish()
        print(f"  Transition complete: alpha reached {alphas[-1]:.4f}")
        print()


def demo_pipeline_transition():
    print("\n" + "="*70)
    print("  Part 2: Pipeline Transition Demo (DTLN->DFN->DTLN)")
    print("="*70)

    config = dataclasses.replace(
        AcousticConfig.development_16khz(),
        dwell_enter=4,
        dwell_exit=4,
        crossfade_frames=6,
        threshold_low=0.30,
        threshold_high=0.55,
    )

    pipeline = AcousticPipeline(config)
    pipeline.set_model_available(ModelID.DEEP_FILTER_NET, True)

    print(f"\n  Thresholds: T_low={config.threshold_low}, T_high={config.threshold_high}")
    print(f"  Dwell enter/exit: {config.dwell_enter}/{config.dwell_exit} frames")
    print(f"  Crossfade frames: {config.crossfade_frames}")
    print(f"  Hop = {1000*config.hop_length/config.sample_rate:.0f} ms -> "
          f"crossfade ~= {config.crossfade_frames * config.hop_length / config.sample_rate * 1000:.0f} ms")
    print()

    print(f"  {'Fr':>4}  {'Phase':<10}  {'C':>6}  {'State':<13}  {'Model':<16}  "
          f"{'Crossfade alpha':<15}  {'Dwell':>5}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*13}  {'-'*16}  {'-'*15}  {'-'*5}")

    frame_idx = 0

    def run(n, amp, label):
        nonlocal frame_idx
        for _ in range(n):
            frame = make_frame(frame_idx, amp=amp)
            diag = pipeline.process(frame)
            rd = diag.router_decision
            state = rd.router_state.name
            model = rd.active_model.value
            alpha = rd.crossfade_alpha
            dwell = rd.dwell_counter
            C = diag.complexity_score

            alpha_bar = int(alpha * 12)
            alpha_str = "#" * alpha_bar + "." * (12 - alpha_bar)
            print(f"  {frame_idx:>4}  {label:<10}  {C:>6.3f}  {state:<13}  "
                  f"{model:<16}  [{alpha_str}]{alpha:.2f}  {dwell:>5}")
            frame_idx += 1

    run(10, 0.005, "quiet")
    run(20, 3.0,   "NOISY")
    run(20, 0.002, "quiet")
    run(20, 3.0,   "NOISY")

    print()
    print(f"  Final active model: {pipeline.active_model.value}")
    print(f"  Final router state: {pipeline.router_state.name}")
    print()
    print("  NOTE: Crossfade alpha during TRANSITION shows the blend factor.")
    print("  Person 4 uses this alpha to blend DTLN and DFN outputs:")
    print("    S_out = (1-alpha)*S_DTLN + alpha*S_DFN")
    print()


def main():
    demo_crossfade_standalone()
    demo_pipeline_transition()
    print("  Demo complete.\n")


if __name__ == "__main__":
    main()
