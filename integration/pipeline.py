"""
pipeline.py
===========
SIH2026 End-to-End Integration Pipeline.

Orchestrates the complete software demonstrator:

    Primary (noisy speech) + Reference (synthetic noise)
        ↓
    Person 2: NLMS Adaptive Noise Cancellation          [nlms.py]
        ↓
    cleaned waveform e[n] + NLMS-derived SNR estimate
        ↓
    P2→P3 buffering via StreamingFramer                 [person3_dsp]
        ↓
    Person 3: STFT (frame-by-frame)                     [person3_dsp]
        ↓
    P5 AcousticFrame construction (one-sided magnitude adaptor)
        ↓
    Person 5: Acoustic Intelligence                     [person5_acoustic]
        ↓  features / complexity C(t) / router decision / crossfade α
    P4 Model Interface                                  [model_interface.py]
        ↓  DTLN slot OR DFN slot (per router decision + crossfade)
    Enhanced complex STFT frame
        ↓
    Person 3: ISTFT + Overlap-Add                       [person3_dsp]
        ↓
    Post-processing safety (NaN/Inf guard, output limiter)
        ↓
    Enhanced output waveform

===========================================================================
ARCHITECTURE OWNERSHIP
===========================================================================

    PERSON 3  owns all STFT/ISTFT/framing/windowing/overlap-add.
              The pipeline MUST NOT introduce another ISTFT, another
              overlap-add, or another synthesis window.

    PERSON 5  owns complexity estimation, router state machine,
              model availability tracking, and crossfade blending.
              The pipeline MUST NOT redesign the router.

    PERSON 4  owns model inference.  The pipeline treats models as
              opaque callables via ModelInterface.process().
              Internal neural-network details are NOT known here.

    PIPELINE  provides only integration / adaptor code and
              fail-safe error handling.

===========================================================================
FAIL-SAFE MODEL HANDLING
===========================================================================

If a model raises an exception or produces invalid output:

    1. The exception is caught here (not in P3 or P5).
    2. A warning is printed.
    3. The model is marked unavailable via P5's set_model_available().
    4. The existing P5 router fallback behaviour takes over.
    5. The audio pipeline continues without crashing.
    6. Invalid spectra are NEVER passed to P3's ISTFT.

===========================================================================
BUFFERING
===========================================================================

    P2 block_size = 1024 samples   (at 16 kHz; scales with sample_rate)
    P3 hop       =  160 samples    (10 ms @ 16 kHz)
    P3 window    =  320 samples    (20 ms @ 16 kHz)

These do NOT align.  Buffering is handled via P3's StreamingFramer.

At 32 kHz:  hop=320, window=640.
At 48 kHz:  hop=480, window=960.
All sizes are derived — nothing is hard-coded.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# --- Person 3 imports (DO NOT replace or duplicate these) ---
from person3_dsp import DSPConfig, STFTProcessor

# --- Person 5 imports (DO NOT replace or duplicate these) ---
from person5_acoustic import (
    AcousticConfig,
    AcousticFrame,
    AcousticPipeline,
    ModelID,
    RouterState,
    PipelineDiagnostics,
    blend,
)

# --- Integration layer ---
from .config import IntegrationConfig
from .nlms import NLMSFilter, NLMSResult
from .model_interface import (
    ModelInterface,
    ModelMetadata,
    ModelOutputError,
    ModelPerformanceStats,
    DTLNStub,
    DFNStub,
)


# ---------------------------------------------------------------------------
# Per-frame result container
# ---------------------------------------------------------------------------

@dataclass
class FrameDiagnostics:
    """Per-frame diagnostics from the full pipeline."""

    frame_index:   int
    timestamp_sec: float

    # P2 NLMS (interpolated over block; available per-block, mapped to frame)
    nlms_snr_db: Optional[float]

    # P5 diagnostics
    rms:                float
    zcr:                float
    spectral_flux:      float
    spectral_entropy:   float
    centroid_variation: float
    transient_score:    float
    estimated_snr_db:   float      # P5 internal proxy estimate
    complexity_score:   float
    router_state:       str
    active_model:       str
    crossfade_alpha:    float
    is_transient:       bool
    dwell_counter:      int

    # Model health for this frame
    model_fallback_used: bool = False
    """True if the active model failed and a fallback was applied."""


@dataclass
class PipelineResult:
    """Full end-to-end pipeline result."""

    # Output audio
    enhanced_wav:     np.ndarray   # final enhanced speech
    nlms_cleaned_wav: np.ndarray   # P2 NLMS output (before STFT/model/ISTFT)
    sample_rate:      int
    output_length:    int

    # NLMS diagnostics
    nlms_result: NLMSResult

    # Per-frame diagnostics
    frame_diagnostics: List[FrameDiagnostics]

    # Model performance
    dtln_stats: ModelPerformanceStats
    dfn_stats:  ModelPerformanceStats

    # Timing
    total_processing_sec: float
    audio_duration_sec:   float

    # Model info (for reporting)
    dtln_metadata: ModelMetadata
    dfn_metadata:  ModelMetadata

    @property
    def real_time_factor(self) -> float:
        return self.total_processing_sec / max(self.audio_duration_sec, 1e-9)

    @property
    def num_frames(self) -> int:
        return len(self.frame_diagnostics)

    @property
    def has_nan(self) -> bool:
        return bool(np.any(np.isnan(self.enhanced_wav)))

    @property
    def has_inf(self) -> bool:
        return bool(np.any(np.isinf(self.enhanced_wav)))

    @property
    def clipping_rate(self) -> float:
        return float(np.mean(np.abs(self.enhanced_wav) >= 1.0))

    @property
    def output_rms_db(self) -> float:
        rms = np.sqrt(np.mean(self.enhanced_wav.astype(np.float64) ** 2))
        return float(20.0 * np.log10(rms + 1e-12))

    @property
    def input_rms_db(self) -> float:
        rms = np.sqrt(np.mean(self.nlms_cleaned_wav.astype(np.float64) ** 2))
        return float(20.0 * np.log10(rms + 1e-12))

    @property
    def fallback_frame_count(self) -> int:
        """Number of frames where a model failure fallback was used."""
        return sum(1 for f in self.frame_diagnostics if f.model_fallback_used)


# ---------------------------------------------------------------------------
# End-to-End Pipeline
# ---------------------------------------------------------------------------

class EndToEndPipeline:
    """
    SIH2026 End-to-End Speech Enhancement Pipeline (software demonstrator).

    Connects:
        P2 (NLMS) → P3 (STFT) → P5 (Router) → P4 (model) → P3 (ISTFT)

    Model slots:
        dtln_model:  Person 4's edge-optimised DTLN variant (or DTLNStub)
        dfn_model:   Person 4's edge-optimised DFN variant  (or DFNStub)

    Real Person 4 models are injected here; no changes to P3 or P5 are
    needed when models are replaced.

    Supported sample rates: 16000, 32000, 48000 Hz.
    (32 kHz and 48 kHz require appropriate audio data.)

    Args:
        config:      IntegrationConfig (all parameters)
        dtln_model:  ModelInterface for DTLN slot  (default: DTLNStub)
        dfn_model:   ModelInterface for DFN slot   (default: DFNStub)
    """

    def __init__(
        self,
        config:      Optional[IntegrationConfig] = None,
        dtln_model:  Optional[ModelInterface]    = None,
        dfn_model:   Optional[ModelInterface]    = None,
    ):
        self.config = config or IntegrationConfig()
        cfg = self.config

        # --- Person 2: NLMS filter ---
        self._nlms = NLMSFilter(cfg.nlms)

        # --- Person 3: DSP ---
        # Sizes are derived from sample_rate + window_ms + hop_ms.
        # Nothing is hard-coded.
        self._dsp_config = DSPConfig(
            sample_rate = cfg.dsp.sample_rate,
            window_ms   = cfg.dsp.window_ms,
            hop_ms      = cfg.dsp.hop_ms,
            tail_policy = cfg.dsp.tail_policy,
            dtype       = cfg.dsp.dtype,
        )
        self._dsp = STFTProcessor(self._dsp_config)

        # --- Person 5: Acoustic pipeline ---
        # fft_size must match P3's n_fft so magnitudes align.
        # AcousticConfig.development_16khz() defaults to fft_size=512;
        # we override to match P3 exactly.
        self._p5_config = AcousticConfig(
            sample_rate  = cfg.dsp.sample_rate,
            frame_length = self._dsp_config.window_length,
            hop_length   = self._dsp_config.hop_length,
            fft_size     = self._dsp_config.n_fft,
        )
        self._p5 = AcousticPipeline(self._p5_config)

        # --- Person 4: Models ---
        self._dtln: ModelInterface = dtln_model or DTLNStub(cfg.dtln_stub_gain)
        self._dfn:  ModelInterface = dfn_model  or DFNStub(cfg.dfn_stub_gain)

        # Validate and register sample-rate compatibility with P5.
        # An incompatible model is unavailable for this pipeline session;
        # it is never silently run at a rate it does not declare.
        sr = cfg.sample_rate
        self._refresh_model_availability(sr)

        # Performance counters (reset with each process() call)
        self._dtln_stats = ModelPerformanceStats()
        self._dfn_stats  = ModelPerformanceStats()

    # ------------------------------------------------------------------
    # Sample-rate validation
    # ------------------------------------------------------------------

    def _refresh_model_availability(self, sample_rate: int) -> None:
        """
        Register model availability for the requested sample rate.

        ModelMetadata is the capability contract.  If a model does not
        declare support for the pipeline sample rate, P5 is told that the
        slot is unavailable so its normal fallback behaviour is used.
        No resampling or silent compatibility override occurs.
        """
        for model_id, model in (
            (ModelID.DTLN, self._dtln),
            (ModelID.DEEP_FILTER_NET, self._dfn),
        ):
            supported = model.supports_sample_rate(sample_rate)
            if not supported:
                print(
                    f"[Pipeline] WARNING: Model {model.name!r} declares "
                    f"supported_sample_rates={model.metadata.supported_sample_rates} "
                    f"but pipeline sample_rate={sample_rate}. "
                    "Marking model unavailable; P5 fallback will be used.",
                    flush=True,
                )
            self._p5.reset_model_health(model_id)
            self._p5.set_model_available(model_id, supported)

    # ------------------------------------------------------------------
    # Session reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all pipeline state for a new audio session.

        Clears NLMS weights, P5 router state, and model recurrent state.
        Models are NOT recreated — only their internal state is cleared.
        """
        self._nlms.reset()
        self._p5.reset()
        self._dtln.reset()
        self._dfn.reset()
        self._dtln_stats.reset()
        self._dfn_stats.reset()

        # Restore capability-based availability and clear transient model
        # health state.  A model that is unsupported at the configured rate
        # remains unavailable after reset.
        self._refresh_model_availability(self.config.sample_rate)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(
        self,
        primary:     np.ndarray,
        reference:   np.ndarray,
        sample_rate: int,
    ) -> PipelineResult:
        """
        Run the complete end-to-end pipeline on primary + reference audio.

        Args:
            primary:     noisy speech (1D float32, normalised ~[-1, 1])
            reference:   noise reference (1D float32, same length)
                         NOTE: current dataset uses a SYNTHETIC reference
                         derived from the noise recording.  Not a real
                         dual-microphone recording.
            sample_rate: audio sample rate (must match config)

        Returns:
            PipelineResult with enhanced_wav, nlms_cleaned_wav, diagnostics
        """
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"process() sample_rate={sample_rate} does not match "
                f"config sample_rate={self.config.sample_rate}. "
                "Reconfigure the pipeline for this rate."
            )

        t_pipeline_start = time.perf_counter()

        primary   = np.asarray(primary,   dtype=np.float32).ravel()
        reference = np.asarray(reference, dtype=np.float32).ravel()

        # Match lengths
        min_len   = min(len(primary), len(reference))
        primary   = primary[:min_len]
        reference = reference[:min_len]

        audio_duration = min_len / sample_rate

        # ----------------------------------------------------------------
        # Phase 1: Person 2 — NLMS Adaptive Noise Cancellation
        # ----------------------------------------------------------------
        print("[P2] Running NLMS adaptive noise cancellation...", flush=True)
        nlms_result = self._nlms.process(primary, reference, sample_rate)
        cleaned     = nlms_result.cleaned_speech   # float32
        nlms_snr_db = nlms_result.nlms_snr_estimate_db

        print(
            f"[P2] NLMS complete: {nlms_result.num_blocks} blocks, "
            f"RTF={nlms_result.real_time_factor:.2f}x, "
            f"NLMS-derived estimated SNR={nlms_snr_db:.1f} dB "
            f"({'WARNING: NaN' if nlms_result.has_nan else 'OK'})",
            flush=True,
        )

        # ----------------------------------------------------------------
        # Phase 2-5: P3 STFT → P5 → P4 → P3 ISTFT (streaming)
        # ----------------------------------------------------------------
        print("[P3/P5/P4] Streaming: STFT → Router → Model → ISTFT...", flush=True)

        frame_diagnostics: List[FrameDiagnostics] = []
        frame_index = 0

        # P3→P5 adaptor: P3 produces a full complex spectrum (n_fft bins).
        # P5 expects a one-sided (real-FFT) magnitude of shape (n_onesided,).
        # This slice is an integration adaptor; P3 is NOT changed.
        n_fft      = self._dsp_config.n_fft
        n_onesided = n_fft // 2 + 1

        # One-sided frequency bins (Hz) for P5
        all_freq_bins       = self._dsp.frequency_bins()
        freq_bins_onesided  = all_freq_bins[:n_onesided].astype(np.float32)

        # P3 streaming framer for P2→P3 buffering
        framer = self._dsp.new_streaming_framer()

        # Person 3 owns the complete ISTFT/overlap-add operation.
        # Integration only collects enhanced spectra and performs one
        # batch inverse after all frames have been processed.
        win_len = self._dsp_config.window_length
        hop_len = self._dsp_config.hop_length
        enhanced_spectra: List[np.ndarray] = []

        # NLMS block → per-block SNR mapping (for passing to P5)
        block_size   = self.config.nlms.block_size
        num_blocks   = nlms_result.num_blocks
        block_snr_map: Dict[int, float] = {}
        for k in range(num_blocks):
            i0, i1 = k * block_size, min((k + 1) * block_size, min_len)
            c_blk   = cleaned[i0:i1]
            n_blk   = nlms_result.estimated_noise[i0:i1]
            sp  = float(np.mean(c_blk.astype(np.float64) ** 2))
            np_ = float(np.mean(n_blk.astype(np.float64) ** 2))
            block_snr_map[k] = 10.0 * np.log10(sp / (np_ + 1e-12))

        # Feed cleaned waveform block-by-block through P3's streaming framer
        for k in range(num_blocks):
            i0, i1    = k * block_size, min((k + 1) * block_size, min_len)
            chunk      = cleaned[i0:i1]
            block_snr  = block_snr_map[k]

            complete_frames = framer.push(chunk)

            for waveform_frame in complete_frames:
                # ---- Person 3: STFT ----
                D = self._dsp.transform_frame(waveform_frame)

                # ---- P3→P5 adaptor: one-sided magnitude ----
                mag_onesided = np.abs(D[:n_onesided]).astype(np.float32)

                # ---- Person 5: Acoustic Intelligence ----
                timestamp = frame_index * hop_len / sample_rate
                p5_frame  = AcousticFrame(
                    frame_index    = frame_index,
                    sample_rate    = sample_rate,
                    waveform       = waveform_frame,
                    magnitude      = mag_onesided,
                    freq_bins      = freq_bins_onesided,
                    timestamp      = timestamp,
                    frame_length   = win_len,
                    hop_length     = hop_len,
                    external_snr_db= block_snr,
                )
                p5_diag: PipelineDiagnostics = self._p5.process(p5_frame)
                decision  = p5_diag.router_decision
                alpha     = decision.crossfade_alpha

                # ---- Person 4: Model inference (fail-safe) ----
                active_model  = decision.active_model
                target_model  = decision.target_model
                router_state  = decision.router_state
                fallback_used = False

                if router_state == RouterState.TRANSITION and target_model is not None:
                    S_out, fallback_used = self._run_transition(
                        active_model, target_model, D, alpha
                    )
                else:
                    S_out, fallback_used = self._run_single_model(active_model, D)

                # ---- Person 3: defer ISTFT / OLA to batch inverse ----
                # The integration layer performs no synthesis, windowing,
                # or overlap-add of its own.
                enhanced_spectra.append(
                    np.asarray(S_out, dtype=np.complex64)
                )

                # Collect per-frame diagnostics
                fv = p5_diag.features
                frame_diagnostics.append(FrameDiagnostics(
                    frame_index          = frame_index,
                    timestamp_sec        = timestamp,
                    nlms_snr_db          = block_snr,
                    rms                  = float(fv.rms)               if fv else 0.0,
                    zcr                  = float(fv.zcr)               if fv else 0.0,
                    spectral_flux        = float(fv.spectral_flux)     if fv else 0.0,
                    spectral_entropy     = float(fv.spectral_entropy)  if fv else 0.0,
                    centroid_variation   = float(fv.centroid_variation)if fv else 0.0,
                    transient_score      = float(fv.transient_score)   if fv else 0.0,
                    estimated_snr_db     = float(fv.estimated_snr_db) if fv else 0.0,
                    complexity_score     = float(p5_diag.complexity_score),
                    router_state         = decision.router_state.name,
                    active_model         = decision.active_model.value,
                    crossfade_alpha      = float(alpha),
                    is_transient          = bool(p5_diag.is_transient),
                    dwell_counter        = int(decision.dwell_counter),
                    model_fallback_used  = fallback_used,
                ))
                frame_index += 1

        # Flush final partial frame from P3
        last_frame = framer.flush()
        if last_frame is not None:
            D       = self._dsp.transform_frame(last_frame)
            S_out, _ = self._run_single_model(self._p5.active_model, D)
            enhanced_spectra.append(
                np.asarray(S_out, dtype=np.complex64)
            )
            frame_index += 1

        # Person 3 owns ISTFT, synthesis-window handling, and overlap-add.
        # Perform exactly one batch inverse over the complete enhanced spectrum.
        if enhanced_spectra:
            S_all = np.stack(enhanced_spectra, axis=0)
            enhanced_wav = self._dsp.inverse(
                S_all,
                output_length=min_len,
            ).astype(np.float32)
        else:
            enhanced_wav = np.zeros(min_len, dtype=np.float32)

        # Post-processing safety:
        #   1. Replace any residual NaN/Inf with 0.0
        enhanced_wav = np.where(np.isfinite(enhanced_wav), enhanced_wav, 0.0)
        #   2. Soft-limit to [-1, 1] to prevent hard clipping
        if self.config.enable_output_limiter:
            enhanced_wav = np.clip(enhanced_wav, -1.0, 1.0)

        t_pipeline_end = time.perf_counter()

        print(
            f"[Pipeline] Complete: {frame_index} frames, "
            f"total time={t_pipeline_end - t_pipeline_start:.2f}s, "
            f"audio={audio_duration:.2f}s, "
            f"RTF={(t_pipeline_end - t_pipeline_start) / audio_duration:.2f}x",
            flush=True,
        )
        if self.fallback_frame_count(frame_diagnostics) > 0:
            print(
                f"[Pipeline] WARNING: {self.fallback_frame_count(frame_diagnostics)} "
                "frame(s) used model fallback (model failure detected).",
                flush=True,
            )

        return PipelineResult(
            enhanced_wav         = enhanced_wav,
            nlms_cleaned_wav     = cleaned,
            sample_rate          = sample_rate,
            output_length        = min_len,
            nlms_result          = nlms_result,
            frame_diagnostics    = frame_diagnostics,
            dtln_stats            = self._dtln_stats,
            dfn_stats             = self._dfn_stats,
            total_processing_sec = t_pipeline_end - t_pipeline_start,
            audio_duration_sec   = audio_duration,
            dtln_metadata        = self._dtln.metadata,
            dfn_metadata         = self._dfn.metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fallback_frame_count(frame_diagnostics: List[FrameDiagnostics]) -> int:
        return sum(1 for f in frame_diagnostics if f.model_fallback_used)

    def _run_single_model(
        self,
        model_id: ModelID,
        D: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Run one model on one STFT frame with fail-safe error handling.

        Returns:
            (S, fallback_used)
            S:            enhanced complex spectrum, same shape as D
            fallback_used: True if the model failed and D was returned as-is
        """
        model, stats = self._resolve_model(model_id)
        if model is None:
            return D.copy().astype(np.complex64), False

        try:
            S = model.process_validated(D, stats)
            return S, False
        except ModelOutputError as exc:
            print(f"[P4] Model failure — {exc}", flush=True)
            self._mark_model_unavailable(model_id)
            return D.copy().astype(np.complex64), True

    def _run_transition(
        self,
        active_model: ModelID,
        target_model: ModelID,
        D: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, bool]:
        """
        Run BOTH models and blend their spectra during a crossfade transition.

        Blend formula (P5 crossfade contract):
            S_out = (1 - alpha) * S_old + alpha * S_new

        Returns:
            (S_blended, fallback_used)
            """
        S_old, fb_old = self._run_single_model(active_model, D)
        S_new, fb_new = self._run_single_model(target_model, D)
        alpha  = float(np.clip(alpha, 0.0, 1.0))
        S_out  = ((1.0 - alpha) * S_old + alpha * S_new).astype(np.complex64)
        return S_out, (fb_old or fb_new)

    def _resolve_model(
        self, model_id: ModelID
    ) -> tuple[Optional[ModelInterface], Optional[ModelPerformanceStats]]:
        """Map ModelID to (ModelInterface, stats)."""
        if model_id == ModelID.DTLN:
            return self._dtln, self._dtln_stats
        elif model_id == ModelID.DEEP_FILTER_NET:
            return self._dfn, self._dfn_stats
        else:
            return None, None

    def _mark_model_unavailable(self, model_id: ModelID) -> None:
        """Mark a model unavailable via P5's existing mechanism."""
        try:
            self._p5.set_model_available(model_id, False)
        except Exception:
            pass

    def _istft_frame(self, S: np.ndarray) -> np.ndarray:
        """
        Legacy single-frame adaptor retained for compatibility with any
        external callers. The main pipeline does NOT use this helper;
        batch synthesis is owned entirely by Person 3's ``inverse()``.
        """
        S_2d = S.reshape(1, -1)
        td = self._dsp.inverse(S_2d)
        win_len = self._dsp_config.window_length
        return td[:win_len].astype(np.float32)

    # ------------------------------------------------------------------
    # Router demo / validation mode
    # ------------------------------------------------------------------

    def run_router_demo(
        self,
        n_frames:    int = 200,
        sample_rate: int = 16000,
    ) -> List[FrameDiagnostics]:
        """
        ROUTER VALIDATION / DEMONSTRATION MODE.

        Runs the P5 router with synthetic white-noise frames to exercise
        the full state machine (STARTUP → DTLN → TRANSITION → DFN → ...).

        CLEARLY LABELLED: This uses SYNTHETIC audio.
        It does NOT represent real speech enhancement output.
        It is for demonstrating and testing the router state machine ONLY.
        """
        print(
            "[ROUTER DEMO] Running router validation/demonstration mode. "
            "SYNTHETIC audio — NOT real speech enhancement output.",
            flush=True,
        )
        demo_p5 = AcousticPipeline(self._p5_config)
        demo_p5.set_model_available(ModelID.DTLN,          True)
        demo_p5.set_model_available(ModelID.DEEP_FILTER_NET, True)

        win_len   = self._dsp_config.window_length
        hop_len   = self._dsp_config.hop_length
        n_fft     = self._dsp_config.n_fft
        n_onesided= n_fft // 2 + 1
        results   = []

        for i in range(n_frames):
            rng      = np.random.default_rng(seed=i)
            waveform = rng.standard_normal(win_len).astype(np.float32) * 0.1
            full_fft = np.fft.fft(waveform)
            mag_onesided = np.abs(full_fft[:n_onesided]).astype(np.float32)

            frame = AcousticFrame(
                frame_index  = i,
                sample_rate  = sample_rate,
                waveform     = waveform,
                magnitude    = mag_onesided,
                frame_length = win_len,
                hop_length   = hop_len,
            )
            diag     = demo_p5.process(frame)
            decision = diag.router_decision
            fv       = diag.features

            results.append(FrameDiagnostics(
                frame_index        = i,
                timestamp_sec      = i * hop_len / sample_rate,
                nlms_snr_db        = None,
                rms                = fv.rms               if fv else 0.0,
                zcr                 = fv.zcr               if fv else 0.0,
                spectral_flux      = fv.spectral_flux     if fv else 0.0,
                spectral_entropy   = fv.spectral_entropy  if fv else 0.0,
                centroid_variation = fv.centroid_variation if fv else 0.0,
                transient_score    = fv.transient_score   if fv else 0.0,
                estimated_snr_db   = fv.estimated_snr_db if fv else 0.0,
                complexity_score   = diag.complexity_score,
                router_state       = decision.router_state.name,
                active_model       = decision.active_model.value,
                crossfade_alpha    = decision.crossfade_alpha,
                is_transient       = diag.is_transient,
                dwell_counter      = decision.dwell_counter,
                model_fallback_used= False,
            ))
        return results
