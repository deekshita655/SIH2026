# INSTRUCTIONS.md — Developer Guide for person3_dsp

Practical, step-by-step guide for installing, testing, and integrating this module. See `README.md` for the full design/architecture writeup.

## 1. Install dependencies

From the `person3_dsp/` directory:

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e .[dev]
```

This installs `numpy`, `scipy`, and `pytest`, and makes `person3_dsp` importable from anywhere in the environment (editable install). Alternatively:

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Run the tests

```bash
pytest tests/ -v
```

Expected: all tests pass (one test is skipped for the silence case, since there's no signal dynamics to check for clicks). If a test fails after you've made changes, re-read the relevant section of `README.md` §20 "Known limitations" first — a few boundary-condition tolerances are intentionally documented rather than "fixed."

## 3. Run the examples

```bash
python examples/demo_stft.py
python examples/demo_reconstruction.py
```

`demo_stft.py` prints the shapes/dtypes you should expect when integrating. `demo_reconstruction.py` prints MSE / max absolute error / reconstruction SNR for a round-trip STFT → ISTFT on a synthetic signal (you should see an SNR in the 70+ dB range).

## 4. Instantiate the processor

```python
from person3_dsp import DSPConfig, STFTProcessor

# Phase 1 (16 kHz)
config = DSPConfig(sample_rate=16000, window_ms=20, hop_ms=10)
dsp = STFTProcessor(config)

# Production (48 kHz) — same class, same code path
config_48k = DSPConfig(sample_rate=48000, window_ms=20, hop_ms=10)
dsp_48k = STFTProcessor(config_48k)
```

`DSPConfig` validates its inputs at construction time (e.g. it will raise `ValueError` immediately if you pass `center=True`, an unsupported `window_type`, or an unsupported `tail_policy`) — you will not get a silent misconfiguration.

## 5. Pass in an audio waveform

The processor expects a **1D, mono, real-valued** NumPy array (float32 or float64, matching `config.dtype`, default float32):

```python
import numpy as np

signal = np.zeros(16000, dtype=np.float32)   # e.g. 1 second of silence at 16 kHz
```

If you have 16-bit PCM integers, convert to float first (this module does not do that conversion for you — it's part of the hardware/ADC layer):

```python
pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
signal = (pcm16.astype(np.float32)) / 32768.0
```

## 6. Obtain D(k, t)

```python
D = dsp.transform(signal)
print(D.shape)   # (num_frames, config.n_fft)  e.g. (100, 320) for 1s @ 16kHz
print(D.dtype)   # complex64 (or complex128 if config.dtype="float64")
```

## 7. Access magnitude / phase / real / imaginary

```python
from person3_dsp import spectrum

mag   = spectrum.magnitude(D)
phase = spectrum.phase(D)          # radians, NOT unwrapped
real  = spectrum.real_part(D)
imag  = spectrum.imag_part(D)

# round-trip sanity check
D_hat = spectrum.reconstruct_complex(mag, phase)   # ≈ D
```

## 8. Reconstruct audio (ISTFT)

```python
reconstructed = dsp.inverse(D, output_length=len(signal))
```

Always pass `output_length=len(original_signal)` if you used `tail_policy="zero_pad"` for the forward transform — otherwise the reconstructed array will include the extra zero-padded tail samples.

If you're reconstructing from a **model-enhanced** spectrum `S` (same shape/dtype as `D`, but with modified content) rather than the original `D`, the call is identical:

```python
S = your_model(D)                 # owned by Person 4, not this module
enhanced = dsp.inverse(S, output_length=len(signal))
```

## 9. Switching from 16 kHz to 48 kHz

Nothing about the API changes — only the `DSPConfig` you construct:

```python
config = DSPConfig(sample_rate=48000, window_ms=20, hop_ms=10)
dsp = STFTProcessor(config)
# config.window_length == 960, config.hop_length == 480, config.n_fft == 960
```

Do **not** hard-code `320` / `160` / `960` / `480` anywhere in code that calls into this module — always read them off `config.window_length` / `config.hop_length` / `config.n_fft`, exactly as this module does internally.

## 10. How another team member should integrate this module

1. Add `person3_dsp` as a local/editable dependency (or vendor the `src/person3_dsp` package) in your own module's environment.
2. Construct one shared `DSPConfig` (sample rate, window/hop ms) that the whole pipeline agrees on, and pass it to every stage that needs it (Person 3's `STFTProcessor`, and whatever Person 4/5 build on top).
3. Call `dsp.transform(signal)` to get `D`, hand `D` (or its magnitude/phase, per your model's needs) to the model layer, get back `S` with the same shape, and call `dsp.inverse(S, output_length=...)` to get the enhanced audio.
4. For real-time/streaming use, use `dsp.new_streaming_framer()` + `dsp.transform_frame(frame)` per chunk instead of `dsp.transform(signal)` on the whole buffer; call `framer.flush()` once at end-of-stream.

## 11. Exact expected shapes

| Call | Shape |
|---|---|
| `dsp.transform(signal)` | `(num_frames, config.n_fft)` |
| `dsp.inverse(D)` (no `output_length`) | `(num_frames - 1) * hop_length + window_length,` |
| `dsp.inverse(D, output_length=N)` | `(N,)` |
| `dsp.transform_frame(frame)` | `(config.n_fft,)` |
| `dsp.frame_times(num_frames)` | `(num_frames,)` |
| `dsp.frequency_bins()` | `(config.n_fft,)` |

`num_frames` for a given signal length can be computed ahead of time via `dsp.num_frames_for_length(len(signal))` (uses `config.tail_policy`).

## 12. Exact expected dtypes

| Quantity | dtype (when `config.dtype="float32"`, the default) |
|---|---|
| Input signal | `float32` (or will be cast) |
| `D` / `S` (complex spectrum) | `complex64` |
| Reconstructed signal | `float32` |
| `frequency_bins()` | `float64` (always, regardless of `config.dtype`) |
| `frame_times()` | `float64` (always) |

If you construct `DSPConfig(dtype="float64")`, the spectrum dtype becomes `complex128` and the time-domain arrays become `float64`.

## 13. Frame indexing & timestamp convention

- Frame `t` covers `signal[t*hop_length : t*hop_length + window_length]`.
- `dsp.frame_times(num_frames)` returns **frame-START** times in seconds (`t * hop_length / sample_rate`) — this is the primary convention used throughout the module. `dsp.frame_center_times(num_frames)` is available if you specifically need center timestamps, but is not the default.

## 14. Tail policy — what to pick

- Use `tail_policy="zero_pad"` (the default) if you don't want to lose any audio at the end of a signal (e.g. offline batch processing of full utterances).
- Use `tail_policy="drop"` if you specifically want to discard a trailing partial frame (e.g. matching some other component in the pipeline that also drops the tail).
- For streaming, this only matters when you call `framer.flush()` — `push()` never applies the tail policy, since it only returns complete frames.

## 15. Common mistakes

- **Hard-coding `320`/`160`.** Always read `config.window_length` / `config.hop_length` / `config.n_fft`; these differ between 16 kHz and 48 kHz.
- **Forgetting `output_length` on `inverse()`.** If you used `tail_policy="zero_pad"`, omitting `output_length` will return a signal longer than your original input (it includes the zero-padded tail frame's contribution).
- **Passing a one-sided (rfft-style) spectrum into `inverse()`.** This module only accepts/returns full complex spectra of shape `(num_frames, n_fft)`. If a model somewhere in the pipeline produces a one-sided spectrum, it must be expanded back to the full complex form (using Hermitian symmetry) before calling `dsp.inverse()`.
- **Assuming `center=True`-style librosa behavior.** This module intentionally does not support centered STFT; frame 0 starts at sample 0, not at `-n_fft//2`.
- **Expecting perfect reconstruction at sample 0.** See README §20 "Known limitations" — the very first sample(s) of any signal have a small, documented, and tested reconstruction error due to the periodic Hann window's `w[0] == 0`.
- **Mixing configs.** Using a different `DSPConfig` for `transform()` than for `inverse()` (e.g. different `hop_ms`) will silently produce a garbled reconstruction — always reuse the exact same `STFTProcessor`/`DSPConfig` for both directions.
