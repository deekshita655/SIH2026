# Person 5: Acoustic Intelligence + Adaptive Model Router

## What is Person 5?

Person 5 is the **Acoustic Intelligence and Adaptive Model Router** module of the Military-Grade Robust Speech Enhancement Architecture. It sits between the DSP/STFT stage (Person 3) and the output/integration stage (Person 4), making real-time decisions about which speech enhancement model to use based on the acoustic complexity of the current environment.

Person 5 is implemented as a **lightweight, deterministic, adaptive signal-analysis** system. It does **not** use neural networks, PyTorch, TensorFlow, or any machine learning training loop.

---

## What Person 5 OWNS

| Responsibility | Module |
|---|---|
| Per-frame acoustic feature extraction | `features.py` |
| Adaptive running statistics (Median, MAD, EWMA) | `adaptive_stats.py` |
| Adaptive normalization (robust, no hard-coded scales) | `normalization.py` |
| Acoustic complexity score C(t) ∈ [0,1] | `complexity.py` |
| Hysteresis + dwell-time logic | `hysteresis.py` |
| Model router state machine | `router.py` |
| Crossfade controller | `crossfade.py` |
| Pipeline orchestration | `pipeline.py` |
| P3/P4 interface contracts | `interfaces.py` |

## What Person 5 does NOT OWN

- STFT / FFT / ISTFT / overlap-add
- DTLN implementation
- DeepFilterNet implementation
- Hardware / ADC / DAC
- NLMS noise estimation
- Person 3's DSP stage
- Person 4's output blending (P5 provides alpha, P4 applies it)

---

## Architecture

```
AcousticFrame (from P3)
       │
       ▼
FeatureExtractor
  ├── RMS
  ├── ZCR
  ├── SpectralFlux
  ├── SpectralEntropy
  ├── SpectralCentroid / CentroidVariation
  ├── TransientScore
  └── EstimatedSNR [PROXY]
       │
       ▼
ComplexityEngine
  ├── FeatureAdaptiveStats (per feature: RunningMAD + 2-timescale EWMA)
  ├── AdaptiveNormalizer (robust z-score / natural bounds)
  └── Weighted sum → C(t) ∈ [0,1]
       │
       ▼
ModelRouter (state machine)
  ├── HysteresisController (T_low, T_high, dwell)
  └── CrossfadeController (alpha ramp)
       │
       ▼
RouterDecision → P4
  (active_model, crossfade_alpha, diagnostics)
```

---

## Feature Vector

```
f_t = [RMS, ZCR, SpectralFlux, SpectralEntropy, CentroidVariation, TransientScore, EstimatedSNR]
```

| Feature | Formula | Bounds | Notes |
|---|---|---|---|
| RMS | `sqrt(mean(x²))` | [0, ∞) | Energy proxy; do NOT use alone for routing |
| ZCR | `(1/(N-1)) * Σ 𝟙(x[n]*x[n-1] < 0)` | [0, 1] | Naturally bounded |
| SpectralFlux | `‖A_t - A_{t-1}‖₂` | [0, ∞) | L2 norm of spectral change |
| SpectralEntropy | `H / log(K)` | [0, 1] | Normalized; flat spectrum → 1 |
| CentroidVariation | `\|SC_t - SC_{t-1}\| / Nyquist` | [0, 1] | Sample-rate aware |
| TransientScore | Ratio-based detector | [0, 1] | No ML required |
| EstimatedSNR | Proxy via noise floor EWMA | (-∞, ∞) dB | **NOT ground truth** |

---

## Adaptive Median / MAD / EWMA

Person 5 uses three complementary adaptive statistics per feature:

### Running Median
- Finite window (default: 31 frames)
- Robust to impulsive noise — a single spike cannot shift the median
- Used for outlier detection and normalization center

### MAD (Median Absolute Deviation)
```
MAD = median(|x_i - median(x)|)
```
- Robust spread estimator (unlike standard deviation)
- Used as the scale in the robust normalization step

### Outlier Detection
```
|x_t - median_t| > k * MAD_t  →  outlier
```
- Default k = 3.0
- Safe handling when MAD ≈ 0 (epsilon floor)

### Thresholded EWMA (Two Timescales)
```
FAST:  alpha = 0.10  → reacts to immediate changes
SLOW:  alpha = 0.01  → tracks long-term environment baseline
```
The SLOW EWMA only updates with non-outlier samples, protecting the baseline from impulsive noise.

---

## Normalization Philosophy

> **No hard-coded feature ranges (e.g., `RMS / 0.5` or `Flux / 10`).**

Instead:

| Feature type | Normalization |
|---|---|
| Naturally bounded [0,1] | Pass-through (ZCR, SpectralEntropy, TransientScore, CentroidVariation) |
| Frequency-dependent | Normalized to Nyquist (sample-rate aware, never hard-coded to 16 kHz) |
| Unbounded / distribution-dependent | Robust z-score: `z = (x - median) / (MAD + ε)` → clipped to [0,1] |
| SNR | Robust z-score then **inverted**: high SNR → low difficulty |

Offline calibration (from development data only) can supply `median` and `MAD` for each feature to replace the live adaptive statistics.

---

## Complexity Score

```
C(t) = w_R·R + w_Z·Z + w_F·F + w_H·H + w_V·V + w_T·T + w_Q·Q ∈ [0, 1]
```

> ⚠️ **The default weights are PLACEHOLDERS for testing only.**
> Final weights MUST be calibrated on a development/validation set.

Current placeholder weights:
```yaml
rms:              0.10
zcr:              0.10
spectral_flux:    0.20
spectral_entropy: 0.15
centroid_var:     0.15
transient:        0.20
snr_difficulty:   0.10
```

---

## Hysteresis

Two thresholds prevent rapid model switching:

```
T_low = 0.35   (switch DFN → DTLN below this)
T_high = 0.65  (switch DTLN → DFN above this)

Zone [T_low, T_high] → HOLD current model
```

> ⚠️ These are INITIAL placeholder values. Must be validated.

---

## Dwell Time

The threshold condition must persist for N consecutive frames before a switch is triggered.

```
dwell_enter = 5 frames × 10 ms/frame = 50 ms  (entering DFN)
dwell_exit  = 5 frames × 10 ms/frame = 50 ms  (returning to DTLN)
```

If the condition breaks before dwell is reached, the counter resets.

---

## Router State Machine

```
STARTUP
  └─[warm-up frames complete]──────────────────→ DTLN

DTLN
  └─[C > T_high for dwell_enter frames]─────────→ TRANSITION
  └─[DFN unavailable]──────────────────────────→ DTLN (stay)

TRANSITION
  └─[crossfade complete]────────────────────────→ DTLN or DFN

DFN
  └─[C < T_low for dwell_exit frames]──────────→ TRANSITION

FALLBACK
  └─[DTLN health restored]─────────────────────→ DTLN
```

Safe behaviors:
- DFN unavailable → stay with DTLN
- Active model fails 3× → enter FALLBACK
- NaN / infinite complexity → hold current state

---

## Crossfade

During transition, Person 4 blends model outputs:

```
S_t = (1 - α) · S_DTLN + α · S_DFN
```

- α ramps from 0.0 → 1.0 over `crossfade_frames` frames (default: 5)
- Supported ramps: **LINEAR** (default), **EQUAL_POWER** (perceptually smoother)
- P5 exposes `blend(old, new, alpha)` utility for P4

---

## Online vs Offline Calibration

| What | When | By whom |
|---|---|---|
| Normalization median/MAD | **Offline** on dev set | Developer |
| Complexity weights | **Offline** via validation experiments | Developer |
| T_low / T_high thresholds | **Offline** via DTLN/DFN quality measurements | Developer |
| Dwell time values | **Offline** via perception tests | Developer |
| Adaptive statistics | **Online** — updated per frame at runtime | P5 automatically |
| Model availability flags | **Runtime** — set when models load/fail | P3/P4 integration |

---

## Why P5 is NOT a Neural Network

Person 5 must run on edge hardware alongside DTLN and DeepFilterNet. A third ML model would:
- Add latency on top of two existing inference passes
- Require offline training data collection for the routing task specifically
- Be opaque and difficult to debug or port to C++

Instead, P5 uses deterministic, mathematically transparent signal analysis:
- Running median and MAD are computable with simple sorting (C++ trivial)
- EWMA is a single multiply-add operation per frame
- The complexity score is a weighted dot product

---

## Expected P3 Interface

P3 supplies one `AcousticFrame` per hop:

```python
@dataclass
class AcousticFrame:
    frame_index: int           # monotonically increasing
    sample_rate: int           # Hz
    waveform: np.ndarray       # shape: (frame_length,)  -- optional
    magnitude: np.ndarray      # |D(k,t)|, shape: (fft_bins,) -- preferred
    complex_stft: np.ndarray   # D(k,t),   optional
    freq_bins: np.ndarray      # Hz,        shape: (fft_bins,)
    timestamp: float           # seconds
    frame_length: int          # samples
    hop_length: int            # samples
    external_snr_db: float     # optional: from P2/NLMS
```

> P5 does NOT recompute the FFT if `magnitude` is already supplied.

---

## Development (16 kHz) Configuration

```python
config = AcousticConfig.development_16khz()
# sample_rate = 16000 Hz
# frame_length = 320 samples (20 ms)
# hop_length   = 160 samples (10 ms)
# fft_size     = 512
```

## Production (48 kHz) Configuration

```python
config = AcousticConfig.production_48khz()
# sample_rate = 48000 Hz
# frame_length = 960 samples (20 ms)
# hop_length   = 480 samples (10 ms)
# fft_size     = 1024
```

All frequency-dependent calculations use `sample_rate` from the config — never hard-coded to 16 kHz.

---

## Installation

```bash
cd person5_acoustic
pip install -e ".[dev]"
```

Or without dev tools:
```bash
pip install -e .
```

---

## Running Tests

```bash
cd person5_acoustic
pytest
```

With coverage:
```bash
pytest --cov=person5_acoustic --cov-report=term-missing
```

---

## Running Examples

```bash
cd person5_acoustic
python examples/demo_features.py
python examples/demo_adaptive_router.py
python examples/demo_transition.py
```

---

## Future P3 / P4 Integration

### Connecting to P3
When P3 is ready:
1. P3 creates `AcousticFrame` per hop and calls `pipeline.process(frame)`
2. P3 supplies `magnitude` (and optionally `complex_stft`, `freq_bins`)
3. P5 will NOT recompute the FFT

### Connecting to P4
P4 receives `RouterDecision` per frame:
```python
decision = diagnostics.router_decision
alpha    = decision.crossfade_alpha
model    = decision.active_model

# P4 blends model outputs:
output = blend(dtln_output, dfn_output, alpha)
```

---

## C++ Portability Considerations

The core algorithms are designed for straightforward C++ translation:

| Algorithm | C++ equivalent |
|---|---|
| RunningMedian | Circular buffer + insertion sort / partial_sort |
| EWMA | Single float + multiply-add per frame |
| OutlierDetector | Circular buffer + median computation |
| HysteresisController | Integer counter + two float comparisons |
| CrossfadeController | Integer counter + linear interpolation |
| ComplexityEngine | Dot product of 7 floats |

Key design choices that aid C++ portability:
- No dynamic allocation in the hot path
- All state is explicit (no hidden closures or dynamic dispatch)
- No SciPy dependencies (only NumPy arithmetic operations)
- Configurable parameters passed at construction (no global state)

---

## What is IMPLEMENTED NOW vs FUTURE

### ✅ Implemented
- All feature extraction (RMS, ZCR, Flux, Entropy, Centroid, Transient, SNR proxy)
- Adaptive statistics (RunningMedian, MAD, EWMA, ThresholdedEWMA)
- Robust normalization (natural bounds + robust z-score)
- Complexity engine with configurable weights
- Hysteresis controller (two-threshold + dwell)
- Router state machine (STARTUP / DTLN / DFN / TRANSITION / FALLBACK)
- Crossfade controller (linear and equal-power ramps)
- Full pipeline orchestration
- Complete test suite (8 test files, 90+ tests)
- Three runnable examples

### 🔲 Future Calibration / Integration
- Final complexity weights (require dev set experiments)
- Final T_low / T_high thresholds (require DTLN vs DFN quality measurements)
- Normalization calibration from real dev data
- Integration with P3 (AcousticFrame supplied by real DSP stage)
- Integration with P4 (RouterDecision consumed by output blending stage)
- External SNR from P2/NLMS noise estimator
- Model performance hints (ModelPerformanceConfig) from offline dev experiments
- Hardware-specific tuning of dwell time / crossfade length
