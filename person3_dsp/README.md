# person3_dsp — Time-Frequency DSP Module

**Owner:** Person 3
**Scope:** Framing → windowing → STFT → (spectrum handed to the model layer) → ISTFT → overlap-add, for the Military-Grade Robust Speech Enhancement Architecture project.

This module is deliberately self-contained. It does **not** implement hardware, the analog front-end, NLMS, DTLN/DeepFilterNet, AGC/limiter, model routing, or dataset/training code — those belong to other people on the team.

---

## 1. What Person 3 owns

- Converting a real time-domain signal into a full-complex spectrogram (`STFT`) and back (`ISTFT`), with correct, explicit overlap-add normalization.
- The canonical spectral representation (`D(k, t)`) and read-only accessors (magnitude / phase / real / imaginary / frequency bins) used by the rest of the pipeline.
- Frame timing and frequency-bin conventions, documented so nobody downstream has to guess.
- Support for both an offline (whole-signal) API and a streaming (chunk-by-chunk) API, using **exactly the same math** in both cases.

Person 3 does **not** own: what happens to the spectrum in between (that's the model layer / Person 4), or higher-level acoustic features computed from it (that's Person 5).

---

## 2. Where this sits in the architecture

```
   Person 2                  Person 3 (this module)                Person 4          Person 3 (this module)
 e_NLMS[n]  ──────▶  framing → Hann window → FFT  ──────▶  D(k,t) ──────▶  model  ──────▶  S(k,t) ──────▶  IFFT → synth window → OLA → normalize  ──────▶  d_AI[n]
```

- Input: `e_NLMS[n]`, a real-valued, mono, time-domain signal from Person 2.
- Output of the forward path: `D(k, t)`, a full complex spectrum.
- The model layer (Person 4) consumes `D(k, t)` (or does its own internal STFT — see §18) and returns an enhanced spectrum `S(k, t)` with the **same shape/dtype convention**.
- Output of the inverse path: `d_AI[n]`, the enhanced real-valued time-domain signal.

---

## 3. Input contract

- Mono, 1D, real-valued NumPy array.
- Internal DSP dtype: `float32` by default (`float64` also supported via `DSPConfig(dtype="float64")`).
- The module does not perform 16-bit PCM ⇄ float conversion itself; callers are expected to hand in already-decoded float samples (typically normalized to `[-1, 1]`) — that conversion is part of the ADC/DAC / hardware layer, which is explicitly out of scope here.

---

## 4. STFT mathematics

```
D(k, t) = sum_{n=0}^{N-1} x[t*H + n] * w[n] * exp(-j*2*pi*k*n/N)
```

- `x`: input signal
- `w`: analysis window (periodic Hann, see §12)
- `N`: `n_fft` (== `window_length` unless explicitly overridden)
- `H`: `hop_length`
- `t`: frame index, `k`: frequency bin index

Implemented as: frame → multiply by window → (optionally zero-pad to `n_fft`) → `numpy.fft.fft` (a **full complex FFT**, never `rfft`).

---

## 5. Phase-1 configuration (16 kHz)

```
sample_rate   = 16000 Hz
window_ms     = 20 ms   →  window_length = 320 samples
hop_ms        = 10 ms   →  hop_length    = 160 samples
n_fft         = 320
overlap       = 50%
frequency resolution = fs / n_fft = 16000 / 320 = 50 Hz
```

`D.shape == (num_frames, 320)`, dtype `complex64`.

**16 kHz: 320 samples = 20 ms, 160 samples = 10 ms.**

---

## 6. Production configuration (48 kHz)

The exact same code path supports:

```
sample_rate   = 48000 Hz
window_ms     = 20 ms   →  window_length = 960 samples
hop_ms        = 10 ms   →  hop_length    = 480 samples
n_fft         = 960
frequency resolution = 48000 / 960 = 50 Hz
```

**48 kHz: 960 samples = 20 ms, 480 samples = 10 ms.**

Nothing in `src/person3_dsp` hard-codes `320` or `160`. All sizes are derived in `config.py` from `sample_rate`, `window_ms`, and `hop_ms`:

```python
window_length = round(sample_rate * window_ms / 1000)
hop_length    = round(sample_rate * hop_ms / 1000)
```

This is verified explicitly in `tests/test_16khz_config.py` and `tests/test_48khz_config.py` (including a regression test that `window_length != 320` at 48 kHz).

---

## 7. Full complex spectrum convention

- `D` (and any spectrum `S` fed back into `inverse()`) is a **full complex spectrum**: shape `(num_frames, n_fft)`, dtype `complex64`/`complex128`. It is never silently reduced to a one-sided (`rfft`-style, `n_fft//2 + 1` bins) spectrum, and phase is never discarded.
- Standard (non-`fftshift`ed) NumPy FFT bin ordering is used: bin `k` corresponds to `k * sample_rate / n_fft` Hz for `k = 0 .. n_fft//2` (DC through Nyquist); bins above `n_fft//2` correspond to negative frequencies (`freq - sample_rate`) under the usual DFT aliasing convention. Bins are **never rearranged** by this module (no implicit `fftshift`).

Accessors (`src/person3_dsp/spectrum.py`):

```python
magnitude = spectrum.magnitude(D)         # abs(D)
phase     = spectrum.phase(D)             # angle(D), NOT unwrapped
real      = spectrum.real_part(D)
imag      = spectrum.imag_part(D)
D_hat     = spectrum.reconstruct_complex(magnitude, phase)   # ≈ D
freqs     = spectrum.frequency_bins(sample_rate, n_fft)      # Hz per bin
```

---

## 8. Frequency-bin convention

```
frequency[k] = k * sample_rate / n_fft
```

For Phase 1 (`sample_rate=16000`, `n_fft=320`): bin spacing is 50 Hz, bin 0 = 0 Hz (DC), bin 160 = 8000 Hz (Nyquist).

---

## 9. Frame timing

Frame **start** timestamps are the primary convention:

```
t_start_samples = frame_index * hop_length
t_start_seconds = t_start_samples / sample_rate
```

For Phase 1: frame 0 starts at 0 ms, frame 1 at 10 ms, frame 2 at 20 ms, etc.

Frame **center** timestamps are also available (`STFTProcessor.frame_center_times`) for convenience (e.g., aligning a spectrogram plot), but are **not** the primary/streaming convention, since a streaming system knows a frame's start time the moment the frame becomes available, while the center time is only meaningful in hindsight.

---

## 10. Non-centered, streaming-compatible framing

Frame `t` covers exactly:

```
x[t * hop_length : t * hop_length + window_length]
```

- No centering (`center=True` is rejected by `DSPConfig` — see `config.py`).
- No implicit/automatic padding of any kind.
- The same framing rule is used for both the offline (`frame_signal`) and streaming (`StreamingFramer`) code paths, so switching between them never changes numerical behavior.

---

## 11. Tail handling

For an offline signal whose length isn't an exact multiple of `hop_length` (relative to `window_length`), `DSPConfig.tail_policy` controls what happens to the leftover samples at the end:

| `tail_policy` | Behavior |
|---|---|
| `"drop"` | Leftover samples that don't complete a full `window_length` frame are discarded. |
| `"zero_pad"` (default) | The trailing partial frame is zero-padded up to `window_length` and processed as one more frame. |

For streaming, `StreamingFramer.push()` only ever returns *complete* frames; call `StreamingFramer.flush()` once, at end-of-stream, to obtain the final (possibly zero-padded, per `tail_policy`) partial frame.

**Note:** because hop-based frame starts advance independently of whether the previous frame was "complete," a signal that is an exact whole number of frames under `tail_policy="drop"` may still produce one *additional* zero-padded frame under `tail_policy="zero_pad"` — the hop-advanced next frame position can fall strictly inside the signal even when the prior frame ended exactly at the signal boundary. This is intentional and covered by `tests/test_boundary_conditions.py`.

---

## 12. Hann window

A **periodic / DFT-even** Hann window is used — the mathematical equivalent of `scipy.signal.windows.hann(N, sym=False)` (with a pure-NumPy fallback in `windowing.py` if SciPy isn't available). This is the mathematically correct choice for OLA-based STFT/ISTFT (as opposed to the symmetric variant, which is intended for FIR filter design and does not satisfy constant-overlap-add at standard hop sizes). The same window is used for both analysis and synthesis. Window generation is isolated in `windowing.py`.

---

## 13. ISTFT

```
S(k,t) → IFFT → synthesis window → overlap-add → normalization → d_AI[n]
```

IFFT frames are **never** concatenated directly — that produces clicks at every hop boundary. Reconstruction always goes through explicit overlap-add (`ola.py`).

---

## 14. Overlap-add normalization

```
signal_accumulator[start:start+N] += frame_time_domain * synthesis_window
window_accumulator[start:start+N] += analysis_window * synthesis_window
output = signal_accumulator / window_accumulator     (window_accumulator floored at eps, never divided by exactly zero)
```

Implemented in `ola.py` and unit-tested independently of the FFT machinery in `tests/test_ola.py`.

---

## 15. Public API

```python
from person3_dsp import DSPConfig, STFTProcessor

config = DSPConfig(
    sample_rate=16000,
    window_ms=20,
    hop_ms=10,
    n_fft=None,          # defaults to window_length
    window_type="hann",
    center=False,        # must be False; centered STFT is not supported
    tail_policy="zero_pad",
)

dsp = STFTProcessor(config)

D = dsp.transform(signal)                              # (num_frames, n_fft) complex64
reconstructed = dsp.inverse(D, output_length=len(signal))  # real, same length as input

# metadata
dsp.frame_times(D.shape[0])      # frame-START timestamps, seconds
dsp.frequency_bins()             # Hz per bin

# streaming
framer = dsp.new_streaming_framer()
frames = framer.push(chunk)      # list of complete window_length-sample frames
for frame in frames:
    Dt = dsp.transform_frame(frame)   # single-frame spectrum, shape (n_fft,)
last = framer.flush()            # None, or final (possibly zero-padded) frame
```

---

## 16. Examples

- `examples/demo_stft.py` — forward STFT, prints shapes/dtypes/config.
- `examples/demo_reconstruction.py` — round-trip STFT → ISTFT, prints MSE / max abs error / reconstruction SNR.

Run with `python examples/demo_stft.py` (see INSTRUCTIONS.md for environment setup).

---

## 17. Running tests

```
pip install -e .[dev]
pytest tests/ -v
```

All 41 tests pass as of this writing (1 skip: silence has no dynamics to check for clicks).

---

## 18. Integration notes for Person 4 (model layer)

- **Do not assume** every model consumes this exact complex spectrum — some models (e.g. DTLN) do their own internal STFT. This module owns the *canonical* DSP representation; model-specific adapters belong outside `person3_dsp`.
- Whatever spectrum you return from your model (`S(k, t)`) must have the same shape (`(num_frames, n_fft)`) and be a full complex array to be passed into `STFTProcessor.inverse()`.
- Use `spectrum.magnitude` / `spectrum.phase` if your model operates on magnitude and reuses the original phase; use `spectrum.reconstruct_complex(mag, phase)` to rebuild a complex spectrum before calling `inverse()`.
- `STFTProcessor.transform_frame()` / `new_streaming_framer()` are there if you need single-frame, low-latency streaming inference rather than whole-signal batch processing.

## 19. Integration notes for Person 5 (acoustic intelligence / features)

- `spectrum.magnitude(D)`, `spectrum.phase(D)`, `spectrum.real_part(D)`, `spectrum.imag_part(D)` give you everything needed for spectral flux, spectral entropy, spectral centroid, centroid variability, transient detection, etc.
- `dsp.frequency_bins()` gives the Hz value of every bin (see §8 for the convention, including how bins above Nyquist map to negative frequencies).
- `dsp.frame_times(num_frames)` gives frame-**start** seconds for aligning features to a timeline; `dsp.frame_center_times(...)` is also available if you'd rather anchor on frame centers.
- Bins are never reordered/fftshifted by this module — if your feature computation wants a signed-frequency axis, apply `np.fft.fftshift` yourself and shift `frequency_bins()` the same way.

---

## 20. Known limitations

- **Leading-edge boundary accuracy.** The periodic Hann window has `w[0] == 0` exactly. For a stream's very first sample, only frame 0 contributes at that position (there is no earlier frame to supply overlap), so reconstruction of the first 1–2 samples of a signal has reduced accuracy relative to the rest of the signal (interior samples reconstruct to numerical/float32 precision). This is inherent to non-centered, streaming-style STFT without look-ahead padding, and is intentional given the streaming requirement (see `tests/test_boundary_conditions.py::test_leading_edge_boundary_is_documented_and_bounded` and `tests/test_stft_istft_identity.py`, which both document and bound this behavior rather than hiding it).
- **Isolated single-frame reconstruction.** A signal exactly `window_length` samples long, with no overlapping neighbor frame, has similarly reduced accuracy near *both* edges (no neighbor to compensate where the window tapers toward zero at either end). Any real multi-frame signal is unaffected beyond the leading edge described above.
- **`n_fft > window_length` (zero-padded FFT).** Supported for the forward transform, but the inverse transform reconstructs by taking only the first `window_length` samples of each IFFT'd frame before overlap-add (matching the spec's OLA definition). This is exact for the required 16 kHz / 48 kHz configurations (where `n_fft == window_length`); it has not been extensively validated for `n_fft > window_length` and should be treated as an approximation if that configuration is ever used.
- Only the Hann window is implemented (Phase 1 requirement); `DSPConfig` validates `window_type` and will reject anything else with a clear error rather than silently falling back.
- No PCM ⇄ float conversion, resampling, or channel down-mixing is performed by this module — inputs must already be mono float.
