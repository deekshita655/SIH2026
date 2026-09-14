# SIH2026 Integration Architecture

## Status Labels

| Component | Status |
|---|---|
| P3 DSP (STFT/ISTFT/OLA) | **VERIFIED** — 41 passed, 1 skipped |
| P5 Acoustic Intelligence + Router | **VERIFIED** — 204 passed |
| P2 NLMS Integration | **VERIFIED** — integration tests passing |
| Integration layer (P2→P3→P5→P4→P3) | **VERIFIED** — 92 integration tests passing (Phase 1 + 2 + 3) |
| P4 DTLN inference | **STUB** — interface-compatible, not a trained model |
| P4 DeepFilterNet inference | **STUB** — interface-compatible, not a trained model |
| Hardware/P1 (microphones, ADC, analog front-end) | **NOT IMPLEMENTED** |
| 16 kHz validation | **ACTIVE** — Phase-1 development dataset (1000 pairs) |
| 32 kHz configuration | **CONFIG ONLY** — no dataset yet |
| 48 kHz configuration | **CONFIG ONLY** — hardware recordings needed |
| Edge deployment (RPi5 / Jetson Orin Nano) | **NOT YET VALIDATED** |

---

## Architecture Overview

```
Primary mic (noisy speech)  +  Reference mic (noise) [hardware: P1]
              |                           |
              +---------> P2 NLMS <-------+
                              |
                    cleaned waveform e[n]
                    NLMS-derived estimated SNR (PROXY — NOT ground truth)
                              |
                    [P2→P3 StreamingFramer buffer]
                    P2 block_size = DERIVED from sample_rate
                    P3 hop/window = DERIVED from sample_rate + window_ms + hop_ms
                    These do NOT align. Explicit buffering required.
                              |
                    P3 StreamingFramer.push(chunk)
                              |
                    Complete window_length-sample frame
                              |
                    P3 STFTProcessor.transform_frame()
                              |
                    D = full complex FFT
                    shape (n_fft,), dtype complex64
                    n_fft = window_length (derived, NOT hard-coded)
                              |
              +---------------+---------------------------+
              |                                           |
    Integration Adapter                          P4 Model Interface
    (one-sided magnitude)                    (full complex spectrum)
              |                                           |
    mag = abs(D[:n_onesided])              model.process_validated(D)
    n_onesided = n_fft//2+1               → catches errors + validates output
              |                            → marks model unavailable on failure
    P5 AcousticFrame(                      → falls back via P5 router mechanism
        magnitude=mag,   # (n_onesided,)  Enhanced spectrum S_out
        waveform=wf,     # (window_length,)(same shape as D)
        external_snr_db=nlms_snr  # PROXY         |
    )                                               |
              |                                     |
    P5 AcousticPipeline.process()                  |
              |                                     |
    PipelineDiagnostics                            |
        complexity C in [0,1]                      |
        RouterDecision                             |
            active_model (DTLN or DFN)            |
            router_state                           |
            crossfade_alpha                        |
              |                                     |
              +-----------> Router decision --------+
              |                                     |
              |         If TRANSITION:              |
              |         S_out = (1-alpha)*S_old + alpha*S_new
              |         Both models run ONLY during crossfade.
              |                                     |
              +---> P3 STFTProcessor.inverse() <----+
                    (uses full complex S_out, shape n_fft)
                              |
                    OLA accumulation buffer
                              |
                    Post-processing: NaN/Inf guard + output limiter [-1,1]
                              |
                    enhanced_wav (float32, same length as input)
```

**Derived sizes at each sample rate (20 ms window / 10 ms hop):**

| Rate | window_length | hop_length | n_fft | n_onesided |
|------|--------------|------------|-------|------------|
| 16 kHz | 320 | 160 | 320 | 161 |
| 32 kHz | 640 | 320 | 640 | 321 |
| 48 kHz | 960 | 480 | 960 | 481 |

**No size is hard-coded anywhere in the integration layer.**

---

## Multi-Rate Support

```python
from integration.config import IntegrationConfig, SUPPORTED_SAMPLE_RATES

# SUPPORTED_SAMPLE_RATES = {16000, 32000, 48000}

# 16 kHz — active development (data available)
config = IntegrationConfig.development_16khz()

# 32 kHz — configuration only (no dataset yet)
config = IntegrationConfig.for_sample_rate(32000)

# 48 kHz — production target (hardware data required)
config = IntegrationConfig.production_48khz()

# Any rate explicitly
config = IntegrationConfig.for_sample_rate(sample_rate=16000)
```

> **Warning**: Do NOT upsample the 16 kHz dataset to claim 32/48 kHz validation.
> 32 kHz and 48 kHz require independent recordings at those rates.

---

## P3 Spectrum Contract

**Owner: Person 3**

| Property | Value |
|---|---|
| Shape | `(n_fft,)` — derived, e.g. `(320,)` at 16 kHz, 20 ms window |
| dtype | `complex64` |
| Convention | Full (two-sided) DFT — NOT one-sided |
| Frequency bins | `k * Fs / N` for k=0..n_fft-1 |
| Normalization | Per DSPConfig (periodic Hann window, no centering) |
| ISTFT input | Must receive full `(n_fft,)` complex frame |

**P3 MUST NOT be changed to produce only `n_fft//2+1` one-sided bins.**

---

## P3 → P5 One-Sided Magnitude Adapter

**Owner: Integration layer (`integration/pipeline.py`)**

P5 expects one-sided (real-FFT) magnitude because:
- Real audio signals have Hermitian symmetry
- Only `N//2+1` bins are unique
- P5 uses `_build_freq_bins(fft_size//2+1)` internally

The adapter (NOT a change to P3):

```python
n_fft      = dsp_config.n_fft        # derived: 320 @ 16k, 640 @ 32k, 960 @ 48k
n_onesided = n_fft // 2 + 1          # 161 / 321 / 481

D = STFTProcessor.transform_frame(waveform)  # (n_fft,) complex64

# P3→P5 adapter — integration layer only, P3 unchanged
mag_onesided  = np.abs(D[:n_onesided]).astype(np.float32)       # (n_onesided,)
freq_onesided = frequency_bins()[:n_onesided].astype(np.float32) # (n_onesided,)

# Full D is preserved for P4 and P3 ISTFT — NOT truncated
```

P5 does NOT independently recompute the FFT.

---

## P2 → P3 Buffering

**Owner: Integration layer via P3's `StreamingFramer`**

| Parameter | Value |
|---|---|
| P2 block size | ~64 ms (derived from sample_rate) |
| P3 window length | `round(sample_rate × 20ms / 1000)` |
| P3 hop length | `round(sample_rate × 10ms / 1000)` |
| Alignment | NOT aligned — explicit buffering required |
| Buffer implementation | `STFTProcessor.new_streaming_framer()` |

```python
framer = STFTProcessor.new_streaming_framer()

for chunk in p2_output_blocks:        # each ~1024 samples @ 16 kHz
    frames = framer.push(chunk)        # returns 0..N complete frames
    for wf in frames:                  # each exactly window_length samples
        D = STFTProcessor.transform_frame(wf)
        ...

last_frame = framer.flush()            # handles remaining samples
```

No samples are silently dropped or duplicated.

---

## P4 Model Interface

**Owner: Integration layer (`integration/model_interface.py`)**

```python
class ModelInterface(ABC):
    @abstractmethod
    def process(self, D: np.ndarray) -> np.ndarray:
        """
        Args:
            D: full complex spectrum, shape (n_fft,), dtype complex64
        Returns:
            S: enhanced complex spectrum, same shape as D, all values finite
        """
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata: ...
    def reset(self) -> None: ...  # called between sessions, NOT between frames
```

**ModelMetadata fields:**

| Field | Purpose |
|---|---|
| `name`, `version` | Identity |
| `supported_sample_rates` | List of `[16000, 32000, 48000]` subsets |
| `is_trained` | `True` only with real trained weights |
| `is_stub` | `True` for development stubs |
| `streaming_causal` | `True` = no future frames needed (required for RT) |
| `lookahead_ms` | 0 = fully causal |
| `backend` | `"CPU"`, `"ONNX"`, `"TFLite"`, `"TorchScript"`, etc. |
| `parameter_count` | Optional, informational |

**Fail-safe model handling:**

```
model.process_validated(D, stats)
    ├─ success → return validated complex64 S, same shape as D
    └─ failure (exception / wrong shape / NaN / Inf / real dtype)
        ├─ stats.failures += 1
        ├─ pipeline marks model unavailable via P5.set_model_available(id, False)
        ├─ P5 router fallback kicks in
        └─ D passed through as-is — pipeline NEVER crashes, NEVER passes bad spectra to P3
```

**How Person 4 replaces stubs:**

```python
from integration.model_interface import ModelInterface, ModelMetadata

class EdgeDTLN(ModelInterface):
    def __init__(self, weights_path: str, sample_rate: int = 16000):
        self._model = load_edge_dtln(weights_path)
        self.reset()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="EdgeDTLN",
            version="1.0.0",
            supported_sample_rates=[16000, 32000],
            is_trained=True,
            is_stub=False,
            streaming_causal=True,
            lookahead_ms=0.0,
            backend="TFLite",
        )

    def process(self, D: np.ndarray) -> np.ndarray:
        return self._model.infer(D)  # Person 4 implements this

    def reset(self) -> None:
        self._model.reset_state()   # clear LSTM hidden state

# Inject into pipeline — no changes to P3, P5, or pipeline logic needed
from integration.pipeline import EndToEndPipeline
pipeline = EndToEndPipeline(
    config=IntegrationConfig.development_16khz(),
    dtln_model=EdgeDTLN("dtln_edge.tflite"),
    dfn_model=EdgeDFN("dfn_edge.tflite"),
)
```

**Current stubs:**

| Stub | Slot | Behavior |
|---|---|---|
| `DTLNStub` | DTLN (low/medium complexity) | Fixed gain 0.95 — NOT noise suppression |
| `DFNStub` | DeepFilterNet (high complexity) | Fixed gain 0.92 — NOT noise suppression |

Both stubs declare `supported_sample_rates=[16000, 32000, 48000]` and `is_stub=True`.

---

## P5 Acoustic Intelligence Interface

**Owner: Person 5 (`person5_acoustic`)**

P5 receives `AcousticFrame` with:
- `magnitude`: one-sided magnitude, shape `(n_onesided,)` — from P3→P5 adapter
- `freq_bins`: one-sided frequency bins, shape `(n_onesided,)` — from P3→P5 adapter
- `waveform`: time-domain frame, shape `(window_length,)` — for ZCR and transient detection
- `external_snr_db`: NLMS-derived PROXY SNR estimate (NOT ground truth)

P5 computes features (RMS, ZCR, spectral flux, entropy, centroid variation,
transient score, estimated SNR) and a complexity score C in [0, 1].

P5 Router uses hysteresis/dwell:
- C > T_high (0.65): switch DTLN → DeepFilterNet after dwell frames
- C < T_low (0.35): switch DeepFilterNet → DTLN after dwell frames
- T_low ≤ C ≤ T_high: keep current model
- During TRANSITION: crossfade `S = (1−α)×S_old + α×S_new`

---

## Metric Taxonomy

Three distinct SNR concepts exist. **They must NOT be conflated:**

| Concept | Source | Use |
|---|---|---|
| **Dataset construction SNR** | `dataset_metadata_1000.csv` field `snr_db` | Ground-truth mixing SNR (−5…+20 dB) |
| **NLMS-derived proxy SNR** | Runtime: `signal_power / noise_power` of NLMS output | Adaptation quality monitor only |
| **Objective enhancement SNR** | `compute_snr(clean, enhanced)` | Actual enhancement metric (needs clean reference) |

**Available metrics:**

| Metric | Library | Availability |
|---|---|---|
| SNR | numpy | Always available |
| SI-SDR | numpy | Always available |
| STOI | pystoi | `None` if not installed |
| PESQ | pesq | `None` if not installed (16 kHz only) |

Install: `pip install pystoi pesq`

---

## Dataset

Location: `d:/SIH2026/`

```
SIH2026/
    clean_speech/           # 160 Common Voice clean speech recordings
    noisy_speech/           # 1000 noisy mixtures (16 kHz, mono, 16-bit PCM)
    noise/
        stationary/         # stationary background noise (5 recordings)
        non_stationary/     # non-stationary noise (1 recording)
        impulsive/          # impulsive noise (5 recordings)
    dataset_metadata_1000.csv
    output/                 # generated by demo_end_to_end.py
        input_noisy_speech.wav
        nlms_cleaned_speech.wav
        enhanced_speech.wav
        frame_diagnostics.csv
        metrics.json
        *.png               # diagnostic plots
```

The dataset is **single-channel only** (no physical reference microphone recordings).
The NLMS reference channel is **synthetically derived** from the noise recordings.

---

## How to Run

### Set up (once)

```powershell
$env:PYTHONUTF8='1'

# Install person3_dsp in editable mode (required)
cd d:\SIH2026\person3_dsp
.venv\Scripts\pip install -e .
cd d:\SIH2026
```

### Run all tests

```powershell
$env:PYTHONUTF8='1'

# P3 DSP tests (42 tests)
cd d:\SIH2026\person3_dsp
..\person3_dsp\.venv\Scripts\python.exe -m pytest tests/ -v

# Integration tests (92 tests — Phase 1, 2, 3)
cd d:\SIH2026
python -m pytest tests/ -v

# Phase 3 tests only (multi-rate, model interface, fail-safe)
python -m pytest tests/test_integration_phase3.py -v
```

### Run end-to-end demo (auto-selects from dataset)

```powershell
$env:PYTHONUTF8='1'
cd d:\SIH2026
python demo_end_to_end.py
```

### Test with your own audio files

```powershell
# Supply your own WAV files (16 kHz, mono recommended)
python demo_end_to_end.py --primary path\to\your_noisy.wav --reference path\to\your_noise.wav

# Choose noise category and SNR from the dataset
python demo_end_to_end.py --noise-category stationary --snr-db 10

# Experiment with NLMS parameters
python demo_end_to_end.py --filter-length 128 --step-size 0.005

# Run router state-machine demo (synthetic frames)
python demo_end_to_end.py --router-demo

# Skip plots (faster)
python demo_end_to_end.py --no-plots
```

### Listen to output

After running the demo, three WAV files are in `output/`:

```
output\input_noisy_speech.wav    ← original noisy input (copy, for comparison)
output\nlms_cleaned_speech.wav   ← after NLMS only (Person 2)
output\enhanced_speech.wav       ← full pipeline output (NLMS + STFT + router + stubs + ISTFT)
```

Listen in any audio player (VLC, Windows Media Player, Audacity):

```powershell
# Windows — open directly
start output\input_noisy_speech.wav
start output\nlms_cleaned_speech.wav
start output\enhanced_speech.wav

# Or open all three in Audacity to compare waveforms side by side
```

> **Note:** Until Person 4 replaces the stubs with trained models, the enhanced output
> will sound similar to the NLMS output — the stubs apply only a small fixed gain and
> perform **no actual noise suppression**.

### Load and process in Python

```python
import sys
sys.path.insert(0, "d:/SIH2026")

from integration.config import IntegrationConfig
from integration.dataset import DatasetLoader
from integration.pipeline import EndToEndPipeline

# --- From dataset ---
loader  = DatasetLoader("d:/SIH2026")
example = loader.select_example(noise_category="stationary", snr_db=5)
primary, reference, fs = loader.load_example(example)

# --- Or from your own WAV files ---
import soundfile as sf
primary,   fs = sf.read("your_noisy.wav",  dtype="float32")
reference, _  = sf.read("your_noise.wav",  dtype="float32")
# Make mono if stereo
if primary.ndim > 1:   primary   = primary.mean(axis=1)
if reference.ndim > 1: reference = reference.mean(axis=1)

# --- Run pipeline ---
config   = IntegrationConfig.development_16khz()   # 16 kHz
pipeline = EndToEndPipeline(config)
result   = pipeline.process(primary, reference, sample_rate=fs)

# --- Listen ---
sf.write("output/enhanced.wav", result.enhanced_wav, fs, subtype="PCM_16")
sf.write("output/nlms.wav",     result.nlms_cleaned_wav, fs, subtype="PCM_16")

# --- Inject Person 4's real models when ready ---
from your_module import EdgeDTLN, EdgeDFN
pipeline = EndToEndPipeline(
    config,
    dtln_model=EdgeDTLN("dtln_edge.tflite"),
    dfn_model=EdgeDFN("dfn_edge.tflite"),
)
```

---

## Known Limitations

1. **P4 models are stubs.** DTLN and DeepFilterNet are NOT trained. The stubs apply small
   fixed gains. No noise suppression is performed until Person 4 provides trained weights.

2. **Single-channel dataset only.** The NLMS reference is synthetically derived from
   the noise recordings, not a real reference microphone channel.

3. **16 kHz development only.** 32 kHz and 48 kHz configuration is supported but no
   dedicated dataset exists at those rates.

4. **No hardware.** P1 (analog front-end, dual microphones, ADC) is not implemented.

5. **Router thresholds uncalibrated.** The complexity thresholds (T_low=0.35, T_high=0.65)
   are placeholders. Calibration requires development data experiments.

6. **NLMS SNR is a proxy.** The NLMS-derived SNR is not ground truth. Use `snr_db` in
   `dataset_metadata_1000.csv` for construction SNR.

7. **Windows console encoding.** Set `$env:PYTHONUTF8='1'` before running any script.

8. **STOI/PESQ not installed by default.** Run `pip install pystoi pesq` to enable them.
   When unavailable, metrics are reported as `"not available"` — never fabricated.

---

## Recommended Next Tasks

1. **Integrate real DTLN model** — subclass `ModelInterface`, implement `process()`, inject
   via `EndToEndPipeline(dtln_model=EdgeDTLN(...))`.
2. **Integrate real DeepFilterNet** — same pattern as above.
3. **Calibrate router thresholds** — run experiments on the development dataset to set
   T_low, T_high, and feature weights.
4. **Compute PESQ/STOI** — install `pystoi` + `pesq` and run `compute_objective_metrics()`
   with `clean_speech/` WAVs as the clean reference.
5. **Compute SI-SDR baseline** — run `compute_si_sdr(clean, noisy)` and
   `compute_si_sdr(clean, enhanced)` to measure improvement.
6. **Collect 32/48 kHz data** — hardware dual-microphone recordings required.
7. **Edge latency measurement** — measure algorithmic latency (framing + OLA) in ms.
8. **Export router to C++** — translate P5 complexity/router logic for embedded target.
