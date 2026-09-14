"""
demo_features.py
================
Demonstrates feature extraction from synthetic audio frames.

Shows how Person 5 extracts the 7-element feature vector from both:
    - waveform-only frames
    - full spectrum frames (as would be supplied by Person 3)

Uses synthetic signals only — no actual audio files required.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    FeatureExtractor,
)


def make_sine_frame(freq_hz: float, sr: int, n: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_noise_frame(n: int, amp: float = 0.2, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amp).astype(np.float32)


def make_silence_frame(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def print_features(label: str, fv):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Frame index:        {fv.frame_index}")
    print(f"  RMS:                {fv.rms:.6f}")
    print(f"  ZCR:                {fv.zcr:.6f}")
    print(f"  Spectral Flux:      {fv.spectral_flux:.6f}")
    print(f"  Spectral Entropy:   {fv.spectral_entropy:.6f}  (H/log(K), in [0,1])")
    print(f"  Spectral Centroid:  {fv.spectral_centroid:.2f} Hz")
    print(f"  Centroid Variation: {fv.centroid_variation:.6f}  (normalized to Nyquist)")
    print(f"  Transient Score:    {fv.transient_score:.6f}")
    print(f"  Estimated SNR:      {fv.estimated_snr_db:.2f} dB  [PROXY - not ground truth]")
    print(f"  Is Transient:       {fv.is_transient}")
    print(f"  Is Zero Energy:     {fv.is_zero_energy}")


def main():
    print("\n" + "="*60)
    print("  Person 5 — Feature Extraction Demo")
    print("="*60)

    # -----------------------------------------------------------------------
    # Configuration: 16 kHz development mode
    # -----------------------------------------------------------------------
    config = AcousticConfig.development_16khz()
    extractor = FeatureExtractor(config)

    sr = config.sample_rate   # 16000
    N  = config.frame_length  # 320 samples = 20 ms
    fft_size = config.fft_size

    def build_frame(idx, waveform, freq_bins=None, magnitude=None):
        mag = magnitude
        if mag is None and waveform is not None:
            mag = np.abs(np.fft.rfft(waveform, n=fft_size)).astype(np.float32)
        freq = freq_bins or np.fft.rfftfreq(fft_size, d=1.0/sr).astype(np.float32)
        return AcousticFrame(
            frame_index=idx,
            sample_rate=sr,
            waveform=waveform,
            magnitude=mag,
            freq_bins=freq,
            frame_length=N,
            hop_length=N // 2,
        )

    # -----------------------------------------------------------------------
    # Frame 1: Silence
    # -----------------------------------------------------------------------
    silence = make_silence_frame(N)
    frame0 = build_frame(0, silence)
    fv0 = extractor.process(frame0)
    print_features("Frame 0: Silence", fv0)

    # -----------------------------------------------------------------------
    # Frame 1: Low-frequency sine (440 Hz)
    # -----------------------------------------------------------------------
    sine_low = make_sine_frame(440, sr, N, amp=0.3)
    frame1 = build_frame(1, sine_low)
    fv1 = extractor.process(frame1)
    print_features("Frame 1: 440 Hz sine, amp=0.3", fv1)

    # -----------------------------------------------------------------------
    # Frame 2: High-frequency sine (7000 Hz)
    # -----------------------------------------------------------------------
    sine_high = make_sine_frame(7000, sr, N, amp=0.3)
    frame2 = build_frame(2, sine_high)
    fv2 = extractor.process(frame2)
    print_features("Frame 2: 7000 Hz sine, amp=0.3", fv2)

    # -----------------------------------------------------------------------
    # Frame 3: White noise
    # -----------------------------------------------------------------------
    noise = make_noise_frame(N, amp=0.2)
    frame3 = build_frame(3, noise)
    fv3 = extractor.process(frame3)
    print_features("Frame 3: White noise, amp=0.2", fv3)

    # -----------------------------------------------------------------------
    # Frame 4: Impulsive spike (transient test)
    # -----------------------------------------------------------------------
    spike = np.zeros(N, dtype=np.float32)
    spike[N // 2] = 2.0  # single large impulse
    frame4 = build_frame(4, spike)
    fv4 = extractor.process(frame4)
    print_features("Frame 4: Impulsive spike (transient expected)", fv4)

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  OBSERVATIONS")
    print("="*60)
    print(f"  Silence     -> ZCR={fv0.zcr:.3f}, Entropy={fv0.spectral_entropy:.3f}")
    print(f"  Low sine    -> ZCR={fv1.zcr:.3f}, Entropy={fv1.spectral_entropy:.3f}")
    print(f"  High sine   -> ZCR={fv2.zcr:.3f}, Entropy={fv2.spectral_entropy:.3f}")
    print(f"  White noise -> ZCR={fv3.zcr:.3f}, Entropy={fv3.spectral_entropy:.3f}")
    print(f"  Spike       -> Transient={fv4.is_transient}, Score={fv4.transient_score:.3f}")

    print("\n  Note: Centroid variation is 0.0 on frame 1 (no previous frame).")
    print("  Note: SNR is a PROXY estimate — not ground truth.")
    print("\n  Demo complete.\n")


if __name__ == "__main__":
    main()
