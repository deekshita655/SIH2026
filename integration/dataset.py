"""
dataset.py
==========
SIH2026 Development Dataset Loader and Example Selector.

Parses dataset_metadata_1000.csv to identify real dataset examples
without hardcoding filenames or fabricating relationships.

Key dataset limitations (from README.txt §8):
  - Single-channel recordings only
  - NO physical dual-microphone reference recordings
  - NLMS reference channel must be synthetically derived from the
    original noise recordings included in noise/stationary/,
    noise/non_stationary/, noise/impulsive/

Development vs Production:
  16 kHz = this dataset (software/development validation)
  48 kHz = production target (requires hardware-recorded data)

Usage:
    loader = DatasetLoader("d:/SIH2026")
    example = loader.select_example(noise_category="stationary", snr_db=5)
    primary, ref, fs = loader.load_example(example)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DatasetExample:
    """
    One noisy mixture example from the SIH2026 development dataset.

    Fields match the dataset_metadata_1000.csv columns.
    Absolute paths are resolved by DatasetLoader.
    """
    file_id: int
    clean_file: str          # basename, e.g. "common_voice_en_19451218.wav"
    noise_file: str          # basename, e.g. "1-22882-A-44.wav"
    noise_category: str      # "stationary", "non_stationary", or "impulsive"
    snr_db: int              # target SNR: -5, 0, 5, 10, 15, 20
    sample_rate: int         # 16000
    channels: int            # 1 (mono)
    output_file: str         # noisy mixture basename
    impulsive_event_start_sec: Optional[float] = None
    impulsive_event_duration_sec: Optional[float] = None

    def __str__(self) -> str:
        return (
            f"DatasetExample(id={self.file_id}, "
            f"noise_category={self.noise_category!r}, "
            f"snr_db={self.snr_db}dB, "
            f"noisy={self.output_file!r})"
        )


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Loads and queries the SIH2026 development dataset.

    Args:
        dataset_root: path to the SIH2026/ directory containing
                      clean_speech/, noisy_speech/, noise/, and
                      dataset_metadata_1000.csv.

    NLMS Reference Channel Note:
        This dataset does NOT contain physical dual-microphone recordings.
        The `load_example()` method derives the reference channel from the
        corresponding original noise file. This is a synthetic reference
        and should be labelled as such — NOT as a real hardware reference.
    """

    METADATA_FILE = "dataset_metadata_1000.csv"
    CLEAN_DIR = "clean_speech"
    NOISY_DIR = "noisy_speech"
    NOISE_DIRS = {
        "stationary":     "noise/stationary",
        "non_stationary": "noise/non_stationary",
        "impulsive":      "noise/impulsive",
    }
    VALID_CATEGORIES = {"stationary", "non_stationary", "impulsive"}
    VALID_SNRS = {-5, 0, 5, 10, 15, 20}

    def __init__(self, dataset_root: str = "."):
        self.root = Path(dataset_root).resolve()
        self._examples: Optional[List[DatasetExample]] = None

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> List[DatasetExample]:
        csv_path = self.root / self.METADATA_FILE
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {csv_path}. "
                f"Expected at {self.root / self.METADATA_FILE}"
            )
        examples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imp_start = None
                imp_dur = None
                if row.get("impulsive_event_start_sec"):
                    try:
                        imp_start = float(row["impulsive_event_start_sec"])
                    except ValueError:
                        pass
                if row.get("impulsive_event_duration_sec"):
                    try:
                        imp_dur = float(row["impulsive_event_duration_sec"])
                    except ValueError:
                        pass
                examples.append(DatasetExample(
                    file_id=int(row["file_id"]),
                    clean_file=row["clean_file"],
                    noise_file=row["noise_file"],
                    noise_category=row["noise_category"],
                    snr_db=int(row["snr_db"]),
                    sample_rate=int(row["sample_rate"]),
                    channels=int(row["channels"]),
                    output_file=row["output_file"],
                    impulsive_event_start_sec=imp_start,
                    impulsive_event_duration_sec=imp_dur,
                ))
        return examples

    @property
    def examples(self) -> List[DatasetExample]:
        if self._examples is None:
            self._examples = self._load_metadata()
        return self._examples

    # ------------------------------------------------------------------
    # Example selection
    # ------------------------------------------------------------------

    def select_example(
        self,
        noise_category: Optional[str] = None,
        snr_db: Optional[int] = None,
        index: int = 0,
    ) -> DatasetExample:
        """
        Select a dataset example by noise category and/or SNR.

        Args:
            noise_category: filter by "stationary", "non_stationary",
                            or "impulsive". None = any.
            snr_db:         filter by target SNR (-5, 0, 5, 10, 15, 20 dB).
                            None = any.
            index:          index within the filtered list (default 0).

        Returns:
            DatasetExample matching the criteria.

        Raises:
            ValueError: if no matching example is found.
        """
        filtered = self.examples

        if noise_category is not None:
            if noise_category not in self.VALID_CATEGORIES:
                raise ValueError(
                    f"noise_category must be one of {self.VALID_CATEGORIES}, "
                    f"got {noise_category!r}"
                )
            filtered = [e for e in filtered if e.noise_category == noise_category]

        if snr_db is not None:
            if snr_db not in self.VALID_SNRS:
                raise ValueError(
                    f"snr_db must be one of {sorted(self.VALID_SNRS)}, "
                    f"got {snr_db}"
                )
            filtered = [e for e in filtered if e.snr_db == snr_db]

        if not filtered:
            raise ValueError(
                f"No dataset examples found for noise_category={noise_category!r}, "
                f"snr_db={snr_db}"
            )

        if index >= len(filtered):
            raise IndexError(
                f"index {index} out of range for {len(filtered)} matching examples"
            )

        return filtered[index]

    # ------------------------------------------------------------------
    # Audio path resolution
    # ------------------------------------------------------------------

    def noisy_path(self, example: DatasetExample) -> Path:
        return self.root / self.NOISY_DIR / example.output_file

    def clean_path(self, example: DatasetExample) -> Path:
        return self.root / self.CLEAN_DIR / example.clean_file

    def noise_path(self, example: DatasetExample) -> Path:
        noise_dir = self.NOISE_DIRS.get(example.noise_category)
        if noise_dir is None:
            raise ValueError(
                f"Unknown noise_category: {example.noise_category!r}"
            )
        return self.root / noise_dir / example.noise_file

    # ------------------------------------------------------------------
    # Audio loading with synthetic reference derivation
    # ------------------------------------------------------------------

    def load_example(
        self,
        example: DatasetExample,
        target_length: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Load primary (noisy speech) and reference (synthetic noise) signals.

        SYNTHETIC REFERENCE CHANNEL:
            The dataset contains no physical dual-microphone recordings.
            The reference channel is derived from the original noise recording
            trimmed/looped to match the primary signal length.
            This is a DEVELOPMENT/SYNTHETIC reference, not a true hardware
            dual-microphone reference channel.

        Returns:
            (primary, reference, sample_rate) as float32 arrays of equal length.
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for audio loading. "
                "Install with: pip install soundfile"
            )

        noisy_p = self.noisy_path(example)
        noise_p = self.noise_path(example)

        if not noisy_p.exists():
            raise FileNotFoundError(f"Noisy file not found: {noisy_p}")
        if not noise_p.exists():
            raise FileNotFoundError(f"Noise file not found: {noise_p}")

        primary, fs_p = sf.read(str(noisy_p), dtype="float32")
        noise_ref, fs_n = sf.read(str(noise_p), dtype="float32")

        # Mono conversion
        if primary.ndim > 1:
            primary = primary.mean(axis=1)
        if noise_ref.ndim > 1:
            noise_ref = noise_ref.mean(axis=1)

        if fs_p != fs_n:
            raise ValueError(
                f"Primary fs={fs_p}, reference fs={fs_n} mismatch. "
                "Cannot synthesise reference."
            )

        fs = fs_p
        min_len = target_length if target_length else len(primary)

        # Loop/tile noise reference to match primary length
        if len(noise_ref) < min_len:
            repeats = int(np.ceil(min_len / len(noise_ref)))
            noise_ref = np.tile(noise_ref, repeats)
        reference = noise_ref[:min_len].astype(np.float32)
        primary = primary[:min_len].astype(np.float32)

        return primary, reference, fs

    # ------------------------------------------------------------------
    # Summary / listing helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return dataset summary statistics."""
        by_cat = {}
        by_snr = {}
        for e in self.examples:
            by_cat[e.noise_category] = by_cat.get(e.noise_category, 0) + 1
            by_snr[e.snr_db] = by_snr.get(e.snr_db, 0) + 1
        return {
            "total": len(self.examples),
            "by_category": by_cat,
            "by_snr_db": dict(sorted(by_snr.items())),
        }
