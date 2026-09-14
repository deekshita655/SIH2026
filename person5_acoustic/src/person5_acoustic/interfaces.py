"""
interfaces.py
=============
Clean dataclass interfaces for Person 5 Acoustic Intelligence Module.

These interfaces define the contract between Person 3 (DSP/STFT) and Person 5
(Acoustic Intelligence + Adaptive Model Router).

Person 5 does NOT compute FFT/STFT - it receives this data from Person 3.
Person 5 does NOT implement DTLN or DeepFilterNet.

Design goals:
- Clean, typed interfaces suitable for later C++ translation
- Support both 16 kHz (dev) and 48 kHz (production) configurations
- Extensible for future P3/P4 integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AcousticConfig:
    """
    Runtime configuration for the P5 pipeline.

    Designed to be sample-rate agnostic. All frequency-dependent values
    are expressed relative to sample_rate and frame_length.

    Development default:  16 kHz, 320-sample frame, 160-sample hop
    Production default:   48 kHz, 960-sample frame, 480-sample hop
    """

    # --- Audio parameters (set by P3 or system) ---------------------------
    sample_rate: int = 16_000          # Hz
    frame_length: int = 320            # samples (20 ms @ 16 kHz)
    hop_length: int = 160              # samples (10 ms @ 16 kHz)

    # --- Feature extraction -----------------------------------------------
    fft_size: int = 512                # FFT bins (should match P3)

    # --- Adaptive statistics windows --------------------------------------
    median_window: int = 31            # frames for running median/MAD
    fast_ewma_alpha: float = 0.10      # fast EWMA (reacts quickly)
    slow_ewma_alpha: float = 0.01      # slow EWMA (tracks long-term baseline)

    # --- Outlier detection ------------------------------------------------
    mad_k: float = 3.0                 # |x - median| > k * MAD → outlier
    mad_epsilon: float = 1e-8          # prevent division by zero

    # --- Normalization ----------------------------------------------------
    norm_epsilon: float = 1e-8         # numerical stability
    norm_clip_sigma: float = 3.0       # robust z-score clip range

    # --- Complexity engine ------------------------------------------------
    complexity_weights: dict = field(default_factory=lambda: {
        "rms":              0.10,
        "zcr":              0.10,
        "spectral_flux":    0.20,
        "spectral_entropy": 0.15,
        "centroid_var":     0.15,
        "transient":        0.20,
        "snr_difficulty":   0.10,
    })
    # NOTE: These are placeholder weights for testing only.
    # Final weights MUST be calibrated using development/validation data.

    # --- Hysteresis thresholds -------------------------------------------
    # NOTE: These are initial placeholder values. Must be calibrated.
    threshold_high: float = 0.65       # switch DTLN → DFN above this
    threshold_low: float = 0.35        # switch DFN → DTLN below this

    # --- Dwell time -------------------------------------------------------
    dwell_enter: int = 5               # frames condition must persist to switch in
    dwell_exit: int = 5                # frames condition must persist to switch out

    # --- Router startup ---------------------------------------------------
    startup_frames: int = 15           # frames in STARTUP before switching to DTLN

    # --- Crossfade --------------------------------------------------------
    crossfade_frames: int = 5          # number of frames for the crossfade ramp

    # --- Transient detector -----------------------------------------------
    transient_rms_ratio: float = 3.0   # instantaneous/local ratio threshold
    transient_flux_ratio: float = 3.0  # flux ratio threshold

    # --- Model performance hints (offline calibration slots) --------------
    # These are NOT populated during runtime. They are filled offline from
    # development measurements of DTLN vs DeepFilterNet quality/cost.
    model_perf: Optional[ModelPerformanceHints] = None

    def __post_init__(self):
        # Validate weights sum to 1.0
        total = sum(self.complexity_weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"complexity_weights must sum to 1.0, got {total:.6f}"
            )
        if any(w < 0 for w in self.complexity_weights.values()):
            raise ValueError("All complexity_weights must be non-negative")
        if self.threshold_low >= self.threshold_high:
            raise ValueError(
                f"threshold_low ({self.threshold_low}) must be < "
                f"threshold_high ({self.threshold_high})"
            )

    @classmethod
    def development_16khz(cls) -> "AcousticConfig":
        """Standard development configuration at 16 kHz."""
        return cls(
            sample_rate=16_000,
            frame_length=320,
            hop_length=160,
            fft_size=512,
        )

    @classmethod
    def production_48khz(cls) -> "AcousticConfig":
        """Standard production configuration at 48 kHz."""
        return cls(
            sample_rate=48_000,
            frame_length=960,
            hop_length=480,
            fft_size=1024,
        )


@dataclass
class ModelPerformanceHints:
    """
    Offline development measurements of model quality and cost.

    IMPORTANT: These values are NOT collected at runtime.
    They come from offline validation on a held-out development set.
    They are used to inform complexity thresholds and cost tradeoffs.

    Fill these after running development experiments:
        python -m person5_acoustic.calibrate --dev-set <path>
    """
    dtln_avg_pesq: float = 0.0         # average PESQ for DTLN on dev set
    dfn_avg_pesq: float = 0.0          # average PESQ for DeepFilterNet
    dtln_rtf: float = 0.0              # real-time factor for DTLN
    dfn_rtf: float = 0.0               # real-time factor for DeepFilterNet
    quality_lambda: float = 1.0        # cost/quality tradeoff weight
    # U = quality - lambda * cost


# ---------------------------------------------------------------------------
# P3 → P5 Frame Interface
# ---------------------------------------------------------------------------

@dataclass
class AcousticFrame:
    """
    Per-frame data package from Person 3 (DSP/STFT stage) to Person 5.

    P5 receives this and must NOT recompute FFT if spectrum data is present.

    All fields except frame_index are optional to allow incremental P3
    integration. P5 will compute what it can from available data.

    Units:
        waveform:       float32 array, amplitude normalized approximately [-1, 1]
        magnitude:      float32 array, non-negative, shape (fft_bins,)
        complex_stft:   complex64 array, shape (fft_bins,)
        freq_bins:      float32 array, Hz values, shape (fft_bins,)
        timestamp:      seconds since session start
        sample_rate:    Hz
        frame_length:   samples
        hop_length:     samples
    """

    frame_index: int                                    # monotonically increasing
    sample_rate: int                                    # Hz

    # Waveform (required for ZCR, transient energy)
    waveform: Optional[np.ndarray] = None               # shape: (frame_length,)

    # Spectral data (preferred - avoids recomputing FFT)
    magnitude: Optional[np.ndarray] = None              # |D(k,t)|, shape: (fft_bins,)
    complex_stft: Optional[np.ndarray] = None           # D(k,t),   shape: (fft_bins,)
    freq_bins: Optional[np.ndarray] = None              # Hz,        shape: (fft_bins,)

    # Metadata
    timestamp: float = 0.0                             # seconds
    frame_length: int = 320                             # samples
    hop_length: int = 160                               # samples

    # Optional: externally computed SNR estimate (e.g., from P2/NLMS)
    external_snr_db: Optional[float] = None            # dB, None if unavailable

    def __post_init__(self):
        if self.waveform is None and self.magnitude is None:
            raise ValueError(
                "AcousticFrame requires at least 'waveform' or 'magnitude'"
            )

    @property
    def has_waveform(self) -> bool:
        return self.waveform is not None

    @property
    def has_spectrum(self) -> bool:
        return self.magnitude is not None

    @property
    def fft_bins(self) -> int:
        if self.magnitude is not None:
            return len(self.magnitude)
        return 0


# ---------------------------------------------------------------------------
# Feature Vector
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    Per-frame acoustic feature vector produced by the feature extraction stage.

    All features are in their raw (un-normalized) form.
    Normalization happens in a separate stage.

    SNR: This is an ESTIMATE / PROXY - NOT ground-truth SNR.
    """

    frame_index: int

    # Raw features
    rms: float = 0.0                   # root mean square energy
    zcr: float = 0.0                   # zero crossing rate [0, 1]
    spectral_flux: float = 0.0         # L2 norm of spectral change
    spectral_entropy: float = 0.0      # normalized spectral entropy [0, 1]
    spectral_centroid: float = 0.0     # Hz, sample-rate aware
    centroid_variation: float = 0.0    # |SC_t - SC_(t-1)| / Nyquist
    transient_score: float = 0.0       # [0, 1] transient/spike indicator
    estimated_snr_db: float = 0.0      # proxy estimate, NOT ground truth

    # Flags
    is_transient: bool = False         # hard transient detection flag
    is_zero_energy: bool = False       # silent/zero-energy frame

    def to_array(self) -> np.ndarray:
        """Return feature values as a numpy array (for vectorized ops)."""
        return np.array([
            self.rms,
            self.zcr,
            self.spectral_flux,
            self.spectral_entropy,
            self.centroid_variation,
            self.transient_score,
            self.estimated_snr_db,
        ], dtype=np.float64)

    def feature_names(self) -> list[str]:
        return [
            "rms", "zcr", "spectral_flux", "spectral_entropy",
            "centroid_variation", "transient_score", "estimated_snr_db",
        ]


@dataclass
class NormalizedFeatureVector:
    """
    Feature vector after adaptive normalization.
    All values are mapped to approximately [0, 1].
    """

    frame_index: int

    rms_norm: float = 0.0
    zcr_norm: float = 0.0
    spectral_flux_norm: float = 0.0
    spectral_entropy_norm: float = 0.0
    centroid_var_norm: float = 0.0
    transient_norm: float = 0.0
    snr_difficulty_norm: float = 0.0   # inverted SNR: high SNR → low difficulty

    # Outlier flags per feature (True = value was detected as outlier)
    outlier_flags: dict = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.rms_norm,
            self.zcr_norm,
            self.spectral_flux_norm,
            self.spectral_entropy_norm,
            self.centroid_var_norm,
            self.transient_norm,
            self.snr_difficulty_norm,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Router / Model Enums
# ---------------------------------------------------------------------------

class ModelID(Enum):
    """Available enhancement models."""
    DTLN = "DTLN"
    DEEP_FILTER_NET = "DeepFilterNet"
    NONE = "None"             # safe fallback / silence


class RouterState(Enum):
    """Explicit states of the model router state machine."""
    DTLN = auto()             # steady-state: DTLN active
    DFN = auto()              # steady-state: DeepFilterNet active
    TRANSITION = auto()       # crossfading between models
    STARTUP = auto()          # initial frames before statistics stabilize
    FALLBACK = auto()         # target model unavailable, using safe fallback


# ---------------------------------------------------------------------------
# Router Decision
# ---------------------------------------------------------------------------

@dataclass
class RouterDecision:
    """
    Per-frame router decision output.

    Exposes enough information for P4 to blend model outputs and for
    diagnostics/logging.
    """

    frame_index: int
    active_model: ModelID               # model whose output is primary
    requested_model: ModelID            # model router wants to switch to
    router_state: RouterState           # current state machine state
    crossfade_alpha: float = 0.0        # 0.0 = fully old model, 1.0 = fully new
    target_model: Optional[ModelID] = None  # during TRANSITION

    # Diagnostics
    complexity_score: float = 0.0
    threshold_high: float = 0.65
    threshold_low: float = 0.35
    dwell_counter: int = 0
    transition_frame: int = 0


# ---------------------------------------------------------------------------
# Full Pipeline Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class PipelineDiagnostics:
    """
    Complete per-frame diagnostic snapshot from the P5 pipeline.
    Used for logging, visualization, and integration testing.
    """

    frame_index: int

    # Feature stage
    features: Optional[FeatureVector] = None
    normalized: Optional[NormalizedFeatureVector] = None

    # Complexity
    complexity_score: float = 0.0
    complexity_contributions: dict = field(default_factory=dict)

    # Router
    router_decision: Optional[RouterDecision] = None

    # Adaptive stats (snapshot)
    fast_ewma_snapshot: dict = field(default_factory=dict)
    slow_ewma_snapshot: dict = field(default_factory=dict)

    # Flags
    is_transient: bool = False
    outlier_flags: dict = field(default_factory=dict)
    model_availability: dict = field(default_factory=dict)
