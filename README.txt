SIH2026 Audio Dataset — Phase 1 Development Dataset
Updated: Phase 3 Integration (v0.2.0)

============================================================
1. AUDIO FORMAT
============================================================
- Sample rate:  16 kHz (development target)
- Channels:     Mono
- Bit depth:    16-bit PCM WAV

============================================================
2. CLEAN SPEECH
============================================================
- 160 Common Voice English speech recordings

============================================================
3. ORIGINAL NOISE DATA
============================================================
- Stationary:     5 recordings
- Non-stationary: 1 recording
- Impulsive:      5 recordings
- Total:         11 recordings

============================================================
4. GENERATED NOISY SPEECH
============================================================
- 1000 noisy speech mixtures
- Target SNRs: -5, 0, 5, 10, 15, 20 dB
- Noise categories: stationary, non_stationary, impulsive

============================================================
5. HOW TO RUN THE PIPELINE (QUICK START)
============================================================

Step 1 — Install person3_dsp (once):
    cd d:\SIH2026\person3_dsp
    .venv\Scripts\pip install -e .
    cd d:\SIH2026

Step 2 — Set UTF-8 output (Windows, every terminal session):
    $env:PYTHONUTF8='1'

Step 3 — Run the end-to-end demo (auto-selects from dataset):
    python demo_end_to_end.py

Step 4 — Listen to output files in output\:
    input_noisy_speech.wav    <- original noisy input
    nlms_cleaned_speech.wav   <- after NLMS (Person 2)
    enhanced_speech.wav       <- full pipeline (NLMS + STFT + model + ISTFT)

    Open any WAV in VLC, Windows Media Player, or Audacity.
    To open from PowerShell:
        start output\input_noisy_speech.wav
        start output\nlms_cleaned_speech.wav
        start output\enhanced_speech.wav

============================================================
6. TEST WITH YOUR OWN AUDIO FILE
============================================================

Supply any 16-kHz mono WAV as primary (noisy speech):
    python demo_end_to_end.py --primary path\to\noisy.wav --reference path\to\noise.wav

Select different dataset examples:
    python demo_end_to_end.py --noise-category stationary --snr-db 10
    python demo_end_to_end.py --noise-category impulsive  --snr-db 0

Experiment with NLMS parameters:
    python demo_end_to_end.py --filter-length 128 --step-size 0.005

Run without plots (faster):
    python demo_end_to_end.py --no-plots

Full argument list:
    python demo_end_to_end.py --help

============================================================
7. RUN ALL TESTS
============================================================

Integration tests (from repo root):
    python -m pytest tests/ -v

P3 DSP tests (from person3_dsp/):
    .venv\Scripts\python.exe -m pytest tests/ -v

Expected results:
    92 integration tests passed  (Phase 1 + 2 + 3)
    42 P3 DSP tests passed

============================================================
8. SUPPORTED SAMPLE RATES
============================================================

16000 Hz  Active development dataset (1000 noisy/clean pairs)
32000 Hz  Configuration support only — no dataset yet
48000 Hz  Production target — requires hardware dual-mic recordings

NOTE: Do NOT upsample the 16-kHz dataset to claim 32/48-kHz validation.

============================================================
9. NLMS / REFERENCE MICROPHONE
============================================================
The current dataset contains single-channel clean/noisy audio.
It does NOT contain synchronized physical dual-microphone recordings.

The NLMS reference channel is SYNTHETICALLY derived from the
original noise recordings. This is clearly documented and NOT
presented as a true hardware dual-microphone reference.

============================================================
10. METRIC TAXONOMY — READ THIS FIRST
============================================================

Three SNR concepts exist. They must NOT be confused:

    (a) Dataset construction SNR
        Source: dataset_metadata_1000.csv column "snr_db"
        Values: -5, 0, 5, 10, 15, 20 dB
        Meaning: target mixing SNR used to generate the noisy file
        NOT a measurement of enhancement quality

    (b) NLMS-derived proxy SNR
        Source: runtime signal/noise power ratio of NLMS output
        Labelled: "NLMS-derived estimated SNR — NOT ground-truth"
        Use: monitor NLMS adaptation quality only
        NOT the same as objective enhancement SNR

    (c) Objective enhancement metrics (require clean reference)
        SNR, SI-SDR:  always available (numpy)
        STOI:         pip install pystoi
        PESQ:         pip install pesq   (16 kHz only)
        Use these to measure actual noise suppression improvement

============================================================
11. P4 MODEL SLOT STATUS
============================================================

Current status: DEVELOPMENT STUBS — NOT trained models

    DTLN slot:         DTLNStub  — fixed gain 0.95, NO noise suppression
    DeepFilterNet slot: DFNStub  — fixed gain 0.92, NO noise suppression

Until Person 4 provides trained model weights, the enhanced output
sounds similar to the NLMS output. The full integration architecture
is in place and ready to receive trained models.

Person 4 injects real models via:
    pipeline = EndToEndPipeline(
        config,
        dtln_model=EdgeDTLN("dtln_edge.tflite"),
        dfn_model=EdgeDFN("dfn_edge.tflite"),
    )

See ARCHITECTURE.md for the full ModelInterface contract.

============================================================
12. DATASET SPLITTING
============================================================
Multiple noisy mixtures may originate from the same clean
speech recording. Training/validation/test splitting MUST be
performed at the speaker/source level, NOT by randomly splitting
all 1000 files.

============================================================
13. IMPULSIVE NOISE
============================================================
Impulsive samples include event start time and event duration
in the metadata CSV. Global SNR and event-local SNR should be
treated as separate evaluation quantities.

============================================================
14. OUTPUT FILES
============================================================
After running demo_end_to_end.py:

    output\
        input_noisy_speech.wav       <- original noisy input
        nlms_cleaned_speech.wav      <- P2 NLMS output
        enhanced_speech.wav          <- full pipeline output
        frame_diagnostics.csv        <- per-frame features, complexity, router
        metrics.json                 <- NLMS, system, audio metrics
        waveforms.png                <- waveform comparison plot
        spectrograms.png             <- spectrogram comparison
        complexity_router.png        <- complexity C(t) + router states
        features.png                 <- acoustic feature traces

============================================================
15. DATASET SCOPE AND LIMITATIONS
============================================================
- The 1000 noisy files are generated mixtures, not 1000 independent
  noise recordings.
- This is a Phase-1 development dataset for initial ML/DSP development.
- It should NOT be presented as the complete final hardware dataset.
- Hardware microphone recordings at 48 kHz are needed for production validation.
