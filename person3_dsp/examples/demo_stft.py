"""
demo_stft.py

Generates a short mono test signal, runs the forward STFT, and prints
the key shape/dtype information a consumer (P4/P5) would need.

Run with:
    python examples/demo_stft.py
"""

import numpy as np

from person3_dsp import DSPConfig, STFTProcessor
from person3_dsp import spectrum


def main():
    sample_rate = 16000
    duration_s = 1.0
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    config = DSPConfig(sample_rate=sample_rate, window_ms=20, hop_ms=10)
    dsp = STFTProcessor(config)

    D = dsp.transform(signal)
    mag = spectrum.magnitude(D)

    print("=== Person 3 DSP: demo_stft ===")
    print(f"sample_rate        : {config.sample_rate} Hz")
    print(f"window_length       : {config.window_length} samples ({config.window_ms} ms)")
    print(f"hop_length          : {config.hop_length} samples ({config.hop_ms} ms)")
    print(f"n_fft               : {config.n_fft}")
    print(f"input signal length : {len(signal)} samples")
    print(f"number of frames    : {D.shape[0]}")
    print(f"number of freq bins : {D.shape[1]}")
    print(f"complex spectrum D shape : {D.shape}")
    print(f"complex spectrum D dtype : {D.dtype}")
    print(f"magnitude shape          : {mag.shape}")
    print(f"first 5 frame-start times (s): {dsp.frame_times(min(5, D.shape[0]))}")
    print(f"first 5 frequency bins (Hz)  : {dsp.frequency_bins()[:5]}")


if __name__ == "__main__":
    main()
