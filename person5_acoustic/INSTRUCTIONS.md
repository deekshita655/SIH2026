# INSTRUCTIONS.md — Person 5 Developer Guide

## 1. How to Install

From the `person5_acoustic/` directory:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with pytest and coverage tools.

Alternatively, for runtime-only (no test tools):

```bash
pip install -e .
```

**Runtime dependencies:** Only `numpy>=1.24`. No PyTorch, TensorFlow, SciPy, or any
ML framework is used or required.

---

## 2. How to Run Tests

```bash
cd person5_acoustic
pytest
```

Verbose output with short tracebacks is the default (see `pyproject.toml`).

With coverage report:

```bash
pytest --cov=person5_acoustic --cov-report=term-missing
```

To run a single test file:

```bash
pytest tests/test_features.py
pytest tests/test_router.py -v
```

---

## 3. How to Run Examples

```bash
python examples/demo_features.py        # Feature extraction from synthetic frames
python examples/demo_adaptive_router.py # Router responding to changing environment
python examples/demo_transition.py      # Crossfade alpha ramp during transition
```

All examples use synthetic signals. No audio files, models, or GPU required.

---

## 4. File-by-File Explanation

### `src/person5_acoustic/`

| File | Purpose |
|---|---|
| `interfaces.py` | Dataclasses: `AcousticConfig`, `AcousticFrame`, `FeatureVector`, `NormalizedFeatureVector`, `RouterDecision`, `PipelineDiagnostics`, `ModelID`, `RouterState` |
| `features.py` | Per-frame feature extraction: RMS, ZCR, SpectralFlux, SpectralEntropy, SpectralCentroid, CentroidVariation, TransientScore, EstimatedSNR |
| `adaptive_stats.py` | Online statistics: `RunningMedian`, `RunningMAD`, `OutlierDetector`, `EWMA`, `ThresholdedEWMA`, `FeatureAdaptiveStats` |
| `normalization.py` | Robust feature normalization: `AdaptiveNormalizer`, `NormalizationCalibration` |
| `complexity.py` | Complexity score engine: `ComplexityEngine`, `ComplexityResult`, `ModelPerformanceConfig` |
| `hysteresis.py` | Two-threshold hysteresis + dwell-time logic: `HysteresisController` |
| `router.py` | Router state machine: `ModelRouter`, `ModelHealth` |
| `crossfade.py` | Crossfade ramp controller: `CrossfadeController`, `blend()` |
| `pipeline.py` | Top-level orchestrator: `AcousticPipeline` |
| `__init__.py` | Public API exports |

### `tests/`

| File | Tests for |
|---|---|
| `test_features.py` | All 7 features, edge cases (silence, first frame, reset, 48kHz) |
| `test_adaptive_stats.py` | RunningMedian, MAD, outlier detection, EWMA, ThresholdedEWMA |
| `test_normalization.py` | Robust z-score, natural bounds, calibration, SNR inversion |
| `test_complexity.py` | Score bounds, weight validation, sensitivity, adaptation |
| `test_hysteresis.py` | Boundary conditions, dwell, no-chatter, DFN↔DTLN |
| `test_router.py` | Startup, DTLN→DFN, DFN→DTLN, unavailable model, fallback, reset |
| `test_crossfade.py` | Alpha 0→1, linear/equal-power, blend formula, lifecycle |
| `test_pipeline.py` | Full integration, state transitions, diagnostics, 48kHz |

### `examples/`

| File | Demonstrates |
|---|---|
| `demo_features.py` | Feature extraction across silence, sine, noise, impulse frames |
| `demo_adaptive_router.py` | Environment transitions: quiet→noisy→quiet with model routing |
| `demo_transition.py` | Crossfade alpha ramp (standalone + in-pipeline) |

---

## 5. How the Adaptive Statistics Work

### Mental model

Think of three layers:

1. **RunningMedian / RunningMAD** — A 31-frame sliding window that always sees
   all data (including outliers). Its job is to give a robust snapshot of the
   *current* window. Outliers affect at most one position in the sorted window.

2. **Fast EWMA** (alpha=0.10) — Tracks the recent signal level closely.
   It reacts within ~10 frames to environmental changes.

3. **Slow EWMA** (alpha=0.01) — Tracks the long-term environmental baseline.
   It only absorbs samples that are NOT classified as outliers (using the MAD gate).
   This protects the baseline from transient spikes.

### How the outlier gate works

Each new sample `x_t` is evaluated:

```
if |x_t - running_median| > k * running_MAD:
    → outlier: update running median/MAD window, but NOT the slow EWMA
else:
    → normal: update everything
```

The result: a brief car horn, gunshot, or other impulse cannot permanently shift the
environmental baseline.

---

## 6. How to Change Configuration

All configuration lives in `AcousticConfig` (`interfaces.py`). Use Python dataclasses
`replace()` to create a modified copy:

```python
import dataclasses
from person5_acoustic import AcousticConfig

config = dataclasses.replace(
    AcousticConfig.development_16khz(),
    threshold_low=0.30,
    threshold_high=0.60,
    dwell_enter=8,
    dwell_exit=6,
    crossfade_frames=10,
    fast_ewma_alpha=0.15,
    slow_ewma_alpha=0.005,
    median_window=51,
)
```

**Do NOT modify `AcousticConfig` fields directly** after creating a pipeline — the
sub-components read config only at construction time.

---

## 7. How to Supply Calibration Data Later

Calibration parameters (median and MAD per feature) are fitted from development data.

```python
from person5_acoustic import FeatureExtractor, AcousticConfig
from person5_acoustic.normalization import NormalizationCalibration

# 1. Extract features from your development set (NOT test set)
config = AcousticConfig.development_16khz()
extractor = FeatureExtractor(config)

dev_features = []
for frame in dev_set:                          # your dev frames
    fv = extractor.process(frame)
    dev_features.append(fv)

# 2. Fit calibration
calibration = NormalizationCalibration.from_dev_features(dev_features)

# 3. Pass to pipeline
pipeline = AcousticPipeline(config, calibration=calibration)
```

**Critical rule:** Only use the development (train/validation) split. Never use the
test set to determine calibration parameters.

Without calibration, the normalizer falls back to live adaptive statistics
(running median/MAD from the current session). This is fine for runtime but
may vary between sessions.

---

## 8. How to Configure Thresholds

Thresholds `T_low` and `T_high` require offline validation.

**Recommended approach:**

1. Run Person 3 + Person 5 on a representative development dataset.
2. Collect `complexity_score` distributions for:
   - "Clean / easy" conditions (target: DTLN should suffice)
   - "Noisy / hard" conditions (target: DFN needed)
3. Choose `T_low` and `T_high` so that:
   - `T_high` separates "hard" from "medium" complexity
   - `T_low` separates "medium" from "easy" complexity
   - The hysteresis zone [T_low, T_high] covers the ambiguous region

**Initial placeholder values (for testing only):**
```
T_low  = 0.35
T_high = 0.65
```

To update:
```python
import dataclasses
config = dataclasses.replace(
    AcousticConfig.development_16khz(),
    threshold_low=0.30,
    threshold_high=0.60,
)
```

---

## 9. How to Configure Dwell Time

Dwell time determines how many **consecutive** frames the threshold condition must
hold before a model switch is triggered.

At 10 ms hop:
- `dwell_enter = 5` → 50 ms persistence required to enter DFN
- `dwell_exit  = 5` → 50 ms persistence required to return to DTLN

**Tuning guidance:**
- Shorter dwell → more responsive, higher risk of chattering
- Longer dwell → more stable, slower to react to genuine changes
- `dwell_enter` (entering expensive model) can be shorter than `dwell_exit`
  to prefer DFN when conditions warrant, but return slowly

```python
import dataclasses
config = dataclasses.replace(
    AcousticConfig.development_16khz(),
    dwell_enter=8,   # 80 ms to enter DFN
    dwell_exit=10,   # 100 ms to return to DTLN
)
```

---

## 10. How the Router Should Eventually Connect to P4

Person 4 (output stage) receives one `RouterDecision` per frame:

```python
diag    = pipeline.process(frame)
decision = diag.router_decision

active_model = decision.active_model       # ModelID.DTLN or ModelID.DEEP_FILTER_NET
alpha        = decision.crossfade_alpha    # 0.0 = fully DTLN, 1.0 = fully DFN
target_model = decision.target_model       # only set during TRANSITION

# P4 requests outputs from both models (if transitioning):
if decision.router_state == RouterState.TRANSITION:
    dtln_output = run_dtln(frame)
    dfn_output  = run_dfn(frame)
    output = blend(dtln_output, dfn_output, alpha)
else:
    if active_model == ModelID.DTLN:
        output = run_dtln(frame)
    else:
        output = run_dfn(frame)
```

The `blend()` utility is exported from `person5_acoustic`:
```python
from person5_acoustic import blend
output = blend(old_model_output, new_model_output, alpha)
```

During non-transition frames, `alpha = 0.0` (no crossfade needed).

---

## 11. How P5 Should Eventually Connect to P3

When Person 3 is ready, the integration is:

```python
# P3 provides one AcousticFrame per hop
frame = AcousticFrame(
    frame_index=idx,
    sample_rate=p3.sample_rate,
    waveform=p3.current_waveform_frame,      # optional but recommended
    magnitude=p3.current_magnitude_spectrum, # preferred: avoids re-FFT
    complex_stft=p3.current_complex_stft,    # optional
    freq_bins=p3.frequency_bin_array,        # optional: P5 can compute from config
    timestamp=p3.current_timestamp,
    frame_length=p3.frame_length,
    hop_length=p3.hop_length,
    external_snr_db=p2.snr_estimate_db,      # optional: from NLMS stage
)

diag = pipeline.process(frame)
```

**Key point:** If `magnitude` is supplied, P5 will NOT recompute the FFT.

**Integration checklist:**
- [ ] P3 creates `AcousticFrame` dataclass per hop
- [ ] P3 supplies `magnitude` (preferred) or `waveform` (minimum)
- [ ] P3 supplies `freq_bins` (or P5 builds them from config)
- [ ] P3 calls `pipeline.set_model_available(ModelID.DEEP_FILTER_NET, True)` when DFN loads
- [ ] P3 calls `pipeline.report_model_failure(model)` if a model crashes
- [ ] P4 reads `diag.router_decision` for alpha and model selection

---

## 12. What Must Be Validated Before Hardware Integration

Before deploying on edge hardware, validate the following:

| Item | How to validate |
|---|---|
| Complexity thresholds T_low / T_high | Run on annotated dev set; measure routing accuracy |
| Complexity weights | Ablation study: vary weights, measure downstream PESQ/STOI |
| Dwell time values | Listen tests + switch-rate measurements |
| Crossfade length | Perceptual evaluation; too short = click, too long = smear |
| Normalization calibration | Verify on held-out conditions (not just dev set) |
| 48 kHz spectral features | Validate with actual 48 kHz recordings |
| Real-time factor | Profile on target hardware (e.g., Raspberry Pi 4) |
| Edge case: persistent noise | Run 10-minute noise session; check for baseline drift |
| Model health / fallback | Simulate model failure; verify fallback triggers |

---

## 13. Common Mistakes to Avoid

### 1. Using test data for calibration
```python
# ❌ WRONG
calibration = NormalizationCalibration.from_dev_features(test_features)

# ✅ CORRECT
calibration = NormalizationCalibration.from_dev_features(dev_train_features)
```

### 2. Hard-coding 16 kHz assumptions
All frequency calculations in P5 use `frame.sample_rate` or `config.sample_rate` —
never a literal `16000`. If you add new features, follow this pattern.

### 3. Reusing extractor state between sessions without reset
```python
# ❌ Each new audio session needs a reset
pipeline.process(frame_from_session_1)
pipeline.process(frame_from_session_2)  # spectral flux uses prev frame!

# ✅ Reset between sessions
pipeline.reset()
pipeline.process(frame_from_session_2)
```

### 4. Treating EstimatedSNR as ground truth
The runtime SNR is a **proxy estimate** only. It does not have access to a clean
reference signal. It should inform the complexity score modestly, not dominate it.

### 5. Setting dwell_enter = 1
This eliminates the protection against single-frame spikes. Always use dwell >= 2.

### 6. Modifying complexity_weights after pipeline construction
The pipeline reads weights at construction. To change weights, create a new
`AcousticConfig` and a new `AcousticPipeline`.

### 7. Calling pipeline.process() on frames out of order
The pipeline maintains temporal state (previous spectrum for flux, EWMA baselines).
Frames must be supplied in order (monotonically increasing `frame_index`).

---

## 14. How to Translate the Core Algorithms to C++ Later

### RunningMedian → C++ circular buffer + std::nth_element

```cpp
// Circular buffer of window_size floats
// On each update: overwrite oldest, call std::nth_element to find median
std::array<float, WINDOW_SIZE> buf;
int head = 0;
buf[head % WINDOW_SIZE] = new_value;
head++;
// Median: copy buf, use std::nth_element(buf.begin(), buf.begin()+N/2, buf.end())
```

### EWMA → single float

```cpp
float ewma_value = 0.0f;
bool initialized = false;

float update(float x, float alpha) {
    if (!initialized) { ewma_value = x; initialized = true; return x; }
    ewma_value = (1.0f - alpha) * ewma_value + alpha * x;
    return ewma_value;
}
```

### HysteresisController → integer counter

```cpp
int dwell_count = 0;
enum Direction { NONE, UP, DOWN } pending = NONE;

Signal update(float complexity) {
    Direction new_dir = NONE;
    if (current_model == DTLN && complexity > T_HIGH) new_dir = UP;
    if (current_model == DFN  && complexity < T_LOW)  new_dir = DOWN;
    if (new_dir != pending) { pending = new_dir; dwell_count = 1; }
    else if (new_dir != NONE) dwell_count++;
    if (pending == UP   && dwell_count >= DWELL_ENTER) return REQUEST_DFN;
    if (pending == DOWN && dwell_count >= DWELL_EXIT)  return REQUEST_DTLN;
    return STAY;
}
```

### CrossfadeController → frame counter + linear interpolation

```cpp
int frame = 0;
int total = CROSSFADE_FRAMES;
float alpha() { return std::min((float)frame / total, 1.0f); }
float step()  { frame++; return alpha(); }
```

### ComplexityEngine → dot product

```cpp
// After normalization (7 floats), weighted sum:
float score = 0.0f;
for (int i = 0; i < 7; i++) score += weights[i] * normalized[i];
score = std::clamp(score, 0.0f, 1.0f);
```

**Translation strategy:**
1. Port `interfaces.py` → C++ structs / POD types
2. Port `adaptive_stats.py` → template classes with fixed-size arrays
3. Port `normalization.py` → single function `normalize_feature(x, median, mad, eps, clip_sigma)`
4. Port `complexity.py` → dot product loop
5. Port `hysteresis.py` → simple integer state machine
6. Port `crossfade.py` → counter + interpolation
7. Port `router.py` → enum state + switch statement
8. Wire together in `pipeline.cpp`

Python tests serve as the ground truth — compare C++ outputs against Python for the
same input sequences to validate the port.
