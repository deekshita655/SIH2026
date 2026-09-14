"""
demo_reconstruction.py

Generates a test waveform, runs STFT -> ISTFT, and prints MSE, max
absolute error, and reconstruction SNR.

Run with:
    python examples/demo_reconstruction.py
"""

import numpy as np

from person3_dsp import DSPConfig, STFTProcessor


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    noise = original.astype(np.float64) - reconstructed.astype(np.float64)
    signal_power = np.mean(original.astype(np.float64) ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(signal_power / (noise_power + 1e-20))


def main():
    sample_rate = 16000
    duration_s = 2.0
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    rng = np.random.default_rng(7)
    signal = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.02 * rng.standard_normal(t.shape[0])
    ).astype(np.float32)

    config = DSPConfig(sample_rate=sample_rate, window_ms=20, hop_ms=10, tail_policy="zero_pad")
    dsp = STFTProcessor(config)

    D = dsp.transform(signal)
    reconstructed = dsp.inverse(D, output_length=len(signal))

    mse = float(np.mean((signal.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    max_abs_err = float(np.max(np.abs(signal.astype(np.float64) - reconstructed.astype(np.float64))))
    snr = snr_db(signal, reconstructed)

    print("=== Person 3 DSP: demo_reconstruction ===")
    print(f"signal length       : {len(signal)} samples ({duration_s} s @ {sample_rate} Hz)")
    print(f"num frames          : {D.shape[0]}")
    print(f"MSE                 : {mse:.3e}")
    print(f"max absolute error  : {max_abs_err:.3e}")
    print(f"reconstruction SNR  : {snr:.2f} dB")


if __name__ == "__main__":
    main()
