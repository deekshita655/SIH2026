#!/usr/bin/env python3
"""
demo_end_to_end.py
==================
SIH2026 End-to-End Speech Enhancement Demonstrator

Runs the complete software pipeline:
  Person 2 (NLMS) -> Person 3 (STFT) -> Person 5 (Router) ->
  Person 4 (stubs) -> Person 3 (ISTFT)

Using REAL audio from the development dataset.

Development status:
  16 kHz = current software/development validation (this demo)
  48 kHz = production target (NOT yet implemented)

NLMS Note:
  The dataset contains single-channel recordings only.
  The reference channel is SYNTHETICALLY derived from the original
  noise recording. This is documented and NOT presented as a true
  dual-microphone hardware reference.

Usage:
    # Auto-select from dataset (stationary noise, 5 dB SNR)
    python demo_end_to_end.py

    # Select by noise type and SNR
    python demo_end_to_end.py --noise-category stationary --snr-db 5

    # Custom audio files
    python demo_end_to_end.py --primary path/to/noisy.wav --reference path/to/noise.wav

    # Experiment with NLMS parameters
    python demo_end_to_end.py --filter-length 128 --step-size 0.005 --block-size 512

    # Run router demo/validation mode in addition to real audio
    python demo_end_to_end.py --router-demo

Full argument list:
    --primary           Path to primary (noisy speech) WAV
    --reference         Path to reference (noise) WAV
    --noise-category    Dataset filter: stationary, non_stationary, impulsive
    --snr-db            Dataset filter: -5, 0, 5, 10, 15, 20
    --dataset-index     Index within filtered examples (default 0)
    --dataset-root      Root of the SIH2026 dataset (default: current dir)
    --sample-rate       Sample rate in Hz (default 16000)
    --filter-length     NLMS filter length (default 64)
    --step-size         NLMS step size μ (default 0.01)
    --block-size        NLMS block size (default 1024)
    --output-dir        Output directory (default: output/)
    --router-demo       Also run router validation/demonstration mode
    --no-plots          Skip plot generation
    --show-plots        Show plots interactively (requires display)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add repository root to path so integration/ and person3_dsp/ are findable
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from integration.config import IntegrationConfig, NLMSConfig, DSPIntegrationConfig
from integration.dataset import DatasetLoader
from integration.pipeline import EndToEndPipeline
from integration.metrics import (
    compute_audio_metrics,
    format_nlms_metrics,
    format_system_metrics,
    format_model_performance,
    save_frame_diagnostics_csv,
    save_metrics_json,
)
from integration import plots as plot_module


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SIH2026 End-to-End Speech Enhancement Demonstrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Audio sources
    p.add_argument("--primary", type=str, default=None,
                   help="Path to primary (noisy speech) WAV file")

    p.add_argument("--reference", type=str, default=None,
                   help="Path to reference noise WAV file (synthetic if omitted)")

    # Dataset selection
    p.add_argument("--noise-category", type=str, default="stationary",
                   choices=["stationary", "non_stationary", "impulsive"],
                   help="Dataset noise category to select")
    p.add_argument("--snr-db", type=int, default=5,
                   choices=[-5, 0, 5, 10, 15, 20],
                   help="Dataset target SNR (dB) to select")
    p.add_argument("--dataset-index", type=int, default=0,
                   help="Index within filtered dataset examples")
    p.add_argument("--dataset-root", type=str, default=str(REPO_ROOT),
                   help="Root directory of the SIH2026 dataset")

    # NLMS parameters (Person 2)
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="Audio sample rate in Hz")
    p.add_argument("--filter-length", type=int, default=64,
                   help="NLMS filter length (taps). Try: 32, 64, 128")
    p.add_argument("--step-size", type=float, default=0.01,
                   help="NLMS step size μ. Try: 0.005, 0.01, 0.05")
    p.add_argument("--block-size", type=int, default=1024,
                   help="NLMS block size (samples). Try: 512, 1024, 2048")

    # Output
    p.add_argument("--output-dir", type=str, default="output",
                   help="Output directory for WAV files, plots, and diagnostics")

    # Demo options
    p.add_argument("--router-demo", action="store_true",
                   help="Also run router validation/demonstration mode")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--show-plots", action="store_true",
                   help="Show plots interactively (requires display)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SIH2026 — Military-Grade Robust Speech Enhancement")
    print("  End-to-End Software Demonstrator")
    print("  Development configuration: 16 kHz mono")
    print("  Supported rates: 16 / 32 / 48 kHz (32 / 48 kHz: config only, no data)")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Build configuration
    # ----------------------------------------------------------------
    nlms_cfg = NLMSConfig(
        filter_length=args.filter_length,
        step_size=args.step_size,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
    )
    dsp_cfg = DSPIntegrationConfig(sample_rate=args.sample_rate)
    config = IntegrationConfig(
        nlms=nlms_cfg,
        dsp=dsp_cfg,
        dataset_root=args.dataset_root,
        output_dir=str(output_dir),
        save_plots=not args.no_plots,
        show_plots=args.show_plots,
    )

    print(f"\n[Config] NLMS: filter_length={args.filter_length}, "
          f"step_size={args.step_size}, block_size={args.block_size}")

    # ----------------------------------------------------------------
    # 2. Load audio
    # ----------------------------------------------------------------
    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed. Run: pip install soundfile")
        sys.exit(1)

    if args.primary and args.reference:
        # Explicit files provided
        print(f"\n[Audio] Loading explicit files:")
        print(f"  Primary:   {args.primary}")
        print(f"  Reference: {args.reference}")
        primary,   fs_p = sf.read(args.primary,   dtype="float32")
        reference, fs_r = sf.read(args.reference, dtype="float32")
        if primary.ndim   > 1: primary   = primary.mean(axis=1)
        if reference.ndim > 1: reference = reference.mean(axis=1)

        # Resample to target rate if the file was recorded at a different rate
        target_sr = args.sample_rate
        if fs_p != target_sr:
            print(f"  Resampling primary:   {fs_p} Hz → {target_sr} Hz ...")
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(target_sr, fs_p)
                primary = resample_poly(primary, target_sr // g, fs_p // g).astype(np.float32)
            except ImportError:
                print("  WARNING: scipy not installed — run: pip install scipy")
                print(f"  Audio will be played at wrong speed ({fs_p} ≠ {target_sr} Hz)")
        if fs_r != target_sr:
            print(f"  Resampling reference: {fs_r} Hz → {target_sr} Hz ...")
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(target_sr, fs_r)
                reference = resample_poly(reference, target_sr // g, fs_r // g).astype(np.float32)
            except ImportError:
                pass  # already warned above

        print(f"  Using sample rate: {target_sr} Hz")

        # Match lengths
        min_len = min(len(primary), len(reference))
        if len(reference) < len(primary):
            # Loop reference
            repeats = int(np.ceil(len(primary) / len(reference)))
            reference = np.tile(reference, repeats)
        primary = primary[:min_len].astype(np.float32)
        reference = reference[:min_len].astype(np.float32)
        dataset_info = {
            "source": "explicit_files",
            "primary": args.primary,
            "reference": args.reference,
        }

    else:
        # Auto-select from dataset
        print(f"\n[Dataset] Loading from {args.dataset_root}")
        print(f"  Selecting: noise_category={args.noise_category!r}, "
              f"snr_db={args.snr_db}dB, index={args.dataset_index}")

        loader = DatasetLoader(args.dataset_root)
        summary = loader.summary()
        print(f"  Dataset summary: {summary['total']} examples — "
              f"{summary['by_category']}")

        try:
            example = loader.select_example(
                noise_category=args.noise_category,
                snr_db=args.snr_db,
                index=args.dataset_index,
            )
        except (ValueError, IndexError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        print(f"  Selected: {example}")
        print(f"  Noisy file:  {loader.noisy_path(example)}")
        print(f"  Noise file:  {loader.noise_path(example)}")
        print(
            f"\n  NOTE: Reference channel is SYNTHETICALLY derived from the "
            f"original noise recording.\n"
            f"  This dataset contains NO physical dual-microphone recordings.\n"
            f"  See README.txt §8 and INTEGRATION.md for details."
        )

        primary, reference, fs = loader.load_example(example)
        dataset_info = {
            "source": "dataset",
            "file_id": example.file_id,
            "clean_file": example.clean_file,
            "noise_file": example.noise_file,
            "noise_category": example.noise_category,
            "snr_db_target": example.snr_db,
            "output_file": example.output_file,
            "reference_type": "SYNTHETIC — derived from noise recording (NOT hardware dual-mic)",
        }

    print(
        f"\n[Audio] Loaded {len(primary)} samples "
        f"({len(primary)/args.sample_rate:.2f}s) at {args.sample_rate} Hz"
    )

    # ----------------------------------------------------------------
    # 3. Run end-to-end pipeline
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Pipeline] Starting end-to-end processing...")
    print("=" * 70)

    pipeline = EndToEndPipeline(config)
    result = pipeline.process(primary, reference, sample_rate=args.sample_rate)

    # ----------------------------------------------------------------
    # 4. Validate output
    # ----------------------------------------------------------------
    print("\n[Validation] Checking output quality...")
    enhanced_metrics = compute_audio_metrics(result.enhanced_wav)
    nlms_metrics_dict = compute_audio_metrics(result.nlms_cleaned_wav)
    primary_metrics = compute_audio_metrics(primary)

    print(f"  Primary RMS:         {primary_metrics['rms']:.6f}")
    print(f"  NLMS cleaned RMS:    {nlms_metrics_dict['rms']:.6f}")
    print(f"  Enhanced RMS:        {enhanced_metrics['rms']:.6f}")
    print(f"  Enhanced peak:       {enhanced_metrics['peak']:.6f}")
    print(f"  Has NaN:             {enhanced_metrics['has_nan']}")
    print(f"  Has Inf:             {enhanced_metrics['has_inf']}")
    print(f"  Is silence:          {enhanced_metrics['is_silence']}")
    print(f"  Output length:       {result.output_length} samples "
          f"({result.output_length/args.sample_rate:.3f}s)")

    if enhanced_metrics["has_nan"] or enhanced_metrics["has_inf"]:
        print("  WARNING: Output contains NaN/Inf — check pipeline configuration")
    if enhanced_metrics["is_silence"]:
        print("  WARNING: Output is silence — check NLMS/OLA configuration")

    # ----------------------------------------------------------------
    # 5. Save audio outputs
    # ----------------------------------------------------------------
    print("\n[Output] Saving WAV files...")

    noisy_wav_path    = output_dir / "input_noisy_speech.wav"
    nlms_wav_path     = output_dir / "nlms_cleaned_speech.wav"
    enhanced_wav_path = output_dir / "enhanced_speech.wav"

    sf.write(str(noisy_wav_path),    primary,                  args.sample_rate, subtype="PCM_16")
    sf.write(str(nlms_wav_path),     result.nlms_cleaned_wav,  args.sample_rate, subtype="PCM_16")
    sf.write(str(enhanced_wav_path), result.enhanced_wav,      args.sample_rate, subtype="PCM_16")

    print(f"  Noisy input:  {noisy_wav_path}")
    print(f"  NLMS cleaned: {nlms_wav_path}")
    print(f"  Enhanced:     {enhanced_wav_path}")

    # ----------------------------------------------------------------
    # 6. Save diagnostics
    # ----------------------------------------------------------------
    print("\n[Diagnostics] Saving frame diagnostics CSV...")
    diag_csv_path = output_dir / "frame_diagnostics.csv"
    save_frame_diagnostics_csv(result.frame_diagnostics, str(diag_csv_path))
    print(f"  Frame diagnostics: {diag_csv_path}")

    # Save metrics JSON
    nlms_m  = format_nlms_metrics(result.nlms_result)
    sys_m   = format_system_metrics(result)
    model_m = format_model_performance(result)
    audio_m = {
        "primary": primary_metrics,
        "nlms_cleaned": nlms_metrics_dict,
        "enhanced": enhanced_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    save_metrics_json(
        nlms_m, sys_m, audio_m, str(metrics_path),
        extra={
            "dataset": dataset_info,
            "nlms_parameters": {
                "filter_length": args.filter_length,
                "step_size": args.step_size,
                "block_size": args.block_size,
                "sample_rate": args.sample_rate,
            },
            "model_performance": model_m,
        }
    )
    print(f"  Metrics JSON:      {metrics_path}")

    # ----------------------------------------------------------------
    # 7. Generate plots
    # ----------------------------------------------------------------
    if not args.no_plots:
        print("\n[Plots] Generating diagnostic plots...")

        frames = result.frame_diagnostics
        timestamps = np.array([f.timestamp_sec for f in frames])
        complexity = np.array([f.complexity_score for f in frames])
        router_states = [f.router_state for f in frames]
        alphas = np.array([f.crossfade_alpha for f in frames])
        rms_vals = np.array([f.rms for f in frames])
        zcr_vals = np.array([f.zcr for f in frames])
        flux_vals = np.array([f.spectral_flux for f in frames])
        entropy_vals = np.array([f.spectral_entropy for f in frames])
        snr_vals = np.array([f.estimated_snr_db for f in frames])

        waveform_plot = output_dir / "waveforms.png"
        plot_module.plot_waveforms(
            primary, result.nlms_cleaned_wav, result.enhanced_wav,
            args.sample_rate, str(waveform_plot), show=args.show_plots,
        )
        print(f"  Waveforms:  {waveform_plot}")

        router_plot = output_dir / "complexity_router.png"
        plot_module.plot_complexity_and_router(
            timestamps, complexity, router_states, alphas,
            str(router_plot), show=args.show_plots,
        )
        print(f"  Router:     {router_plot}")

        spectrogram_plot = output_dir / "spectrograms.png"
        plot_module.plot_spectrograms(
            primary, result.nlms_cleaned_wav, result.enhanced_wav,
            args.sample_rate, str(spectrogram_plot), show=args.show_plots,
        )
        print(f"  Spectrograms: {spectrogram_plot}")

        features_plot = output_dir / "features.png"
        plot_module.plot_features(
            timestamps, rms_vals, zcr_vals, flux_vals, entropy_vals, snr_vals,
            str(features_plot), show=args.show_plots,
        )
        print(f"  Features:   {features_plot}")

    # ----------------------------------------------------------------
    # 8. Router demo mode
    # ----------------------------------------------------------------
    if args.router_demo:
        print("\n" + "=" * 70)
        print("[Router Demo] VALIDATION / DEMONSTRATION MODE")
        print("  SYNTHETIC complexity ramp — NOT real audio processing")
        print("=" * 70)
        demo_diags = pipeline.run_router_demo(n_frames=200, sample_rate=args.sample_rate)

        demo_timestamps = np.array([d.timestamp_sec for d in demo_diags])
        demo_complexity = np.array([d.complexity_score for d in demo_diags])
        demo_states = [d.router_state for d in demo_diags]
        demo_alphas = np.array([d.crossfade_alpha for d in demo_diags])

        if not args.no_plots:
            router_demo_plot = output_dir / "router_demo_validation.png"
            plot_module.plot_complexity_and_router(
                demo_timestamps, demo_complexity, demo_states, demo_alphas,
                str(router_demo_plot), show=args.show_plots,
            )
            print(f"\n  Router demo plot: {router_demo_plot}")

        state_counts = {}
        for d in demo_diags:
            state_counts[d.router_state] = state_counts.get(d.router_state, 0) + 1
        print(f"\n  Router demo states: {state_counts}")

    # ----------------------------------------------------------------
    # 9. Final summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Audio duration:      {result.audio_duration_sec:.2f}s")
    print(f"  Frames processed:    {result.num_frames}")
    print(f"  Processing time:     {result.total_processing_sec:.2f}s")
    print(f"  Real-time factor:    {result.real_time_factor:.2f}x")
    print(f"\n  NLMS-derived estimated SNR: {result.nlms_result.nlms_snr_estimate_db:.1f} dB")
    print(f"  (This is a reference-based estimate, NOT ground-truth SNR)")

    if frames := result.frame_diagnostics:
        c_vals = [f.complexity_score for f in frames]
        print(f"\n  Complexity C(t): mean={np.mean(c_vals):.3f}, "
              f"min={np.min(c_vals):.3f}, max={np.max(c_vals):.3f}")
        state_counts = {}
        for f in frames:
            state_counts[f.router_state] = state_counts.get(f.router_state, 0) + 1
        print(f"  Router states: {state_counts}")

    print(f"\n  Output WAV files:")
    print(f"    Noisy input:   {noisy_wav_path}")
    print(f"    NLMS cleaned:  {nlms_wav_path}")
    print(f"    Enhanced:      {enhanced_wav_path}")
    print(f"\n  Diagnostics:")
    print(f"    Metrics JSON:  {metrics_path}")
    print(f"    Frame CSV:     {diag_csv_path}")
    if not args.no_plots:
        print(f"    Plots:         {output_dir}/*.png")

    print("\n" + "=" * 70)
    print("  P4 MODEL SLOT STATUS")
    print("=" * 70)
    dtln_md = result.dtln_metadata
    dfn_md  = result.dfn_metadata
    dtln_tag = "✅ TRAINED" if dtln_md.is_trained else "⏳ DEVELOPMENT STUB — no noise suppression"
    dfn_tag  = "✅ TRAINED" if dfn_md.is_trained  else "⏳ DEVELOPMENT STUB — no noise suppression"
    print(f"  DTLN slot:          {dtln_md.name}")
    print(f"    Status:           {dtln_tag}")
    print(f"    Backend:          {dtln_md.backend}")
    print(f"    Frames run:       {result.dtln_stats.frames_processed}")
    print(f"  DeepFilterNet slot: {dfn_md.name}")
    print(f"    Status:           {dfn_tag}")
    print(f"    Backend:          {dfn_md.backend}")
    print(f"    Frames run:       {result.dfn_stats.frames_processed}")
    if result.fallback_frame_count > 0:
        print(f"  ⚠️  Model fallback used: {result.fallback_frame_count} frame(s)")

    print("\n" + "=" * 70)
    print("  COMPONENT STATUS")
    print("=" * 70)
    print("  ✅  Person 2 NLMS — Python dev impl (P2 MATLAB reference port)")
    print("  ✅  Person 3 DSP (STFT/ISTFT) — validated, owns framing/OLA")
    print("  ✅  Person 5 Acoustic Intelligence — validated, owns router")
    print(f"  ⏳  Person 4 DTLN — {dtln_tag}")
    print(f"  ⏳  Person 4 DeepFilterNet — {dfn_tag}")
    print("  ⏳  32/48 kHz — configuration support only (no data yet)")
    print("  ⏳  Edge deployment (RPi5/Jetson) — NOT yet validated")
    print()
    print("  To inject Person 4 real models:")
    print("    pipeline = EndToEndPipeline(config, dtln_model=YourDTLN(), dfn_model=YourDFN())")
    print("=" * 70)


if __name__ == "__main__":
    main()
