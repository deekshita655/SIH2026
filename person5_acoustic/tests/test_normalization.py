"""
test_normalization.py
=====================
Tests for person5_acoustic.normalization module.
"""

import pytest
import numpy as np

from person5_acoustic import AcousticConfig, FeatureVector
from person5_acoustic.normalization import (
    AdaptiveNormalizer,
    NormalizationCalibration,
    FeatureCalibration,
    _robust_normalize,
    _sigmoid_normalize,
    _clip01,
)
from person5_acoustic.adaptive_stats import AdaptiveStatsSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_snap(is_outlier: bool = False) -> AdaptiveStatsSnapshot:
    return AdaptiveStatsSnapshot(
        median=0.0, mad=1.0, fast_ewma=0.0, slow_ewma=0.0,
        is_outlier=is_outlier,
    )


def _snap(median: float, mad: float, is_outlier: bool = False) -> AdaptiveStatsSnapshot:
    return AdaptiveStatsSnapshot(
        median=median, mad=mad, fast_ewma=median, slow_ewma=median,
        is_outlier=is_outlier,
    )


def _feature_vec(
    rms: float = 0.1,
    zcr: float = 0.3,
    flux: float = 0.5,
    entropy: float = 0.7,
    centroid_var: float = 0.1,
    transient: float = 0.0,
    snr_db: float = 10.0,
) -> FeatureVector:
    fv = FeatureVector(frame_index=0)
    fv.rms = rms
    fv.zcr = zcr
    fv.spectral_flux = flux
    fv.spectral_entropy = entropy
    fv.centroid_variation = centroid_var
    fv.transient_score = transient
    fv.estimated_snr_db = snr_db
    return fv


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestClip01:
    def test_below_zero_clipped(self):
        assert _clip01(-1.0) == pytest.approx(0.0)

    def test_above_one_clipped(self):
        assert _clip01(2.0) == pytest.approx(1.0)

    def test_zero_preserved(self):
        assert _clip01(0.0) == pytest.approx(0.0)

    def test_one_preserved(self):
        assert _clip01(1.0) == pytest.approx(1.0)

    def test_midpoint_preserved(self):
        assert _clip01(0.5) == pytest.approx(0.5)


class TestRobustNormalize:
    def test_at_median_maps_to_half(self):
        # z=0 → (0 + clip) / (2*clip) = 0.5
        v = _robust_normalize(5.0, median=5.0, mad=1.0, clip_sigma=3.0)
        assert v == pytest.approx(0.5)

    def test_outlier_clipped_to_one(self):
        # Very high value → clipped to 1.0
        v = _robust_normalize(1e9, median=0.0, mad=1.0, clip_sigma=3.0)
        assert v == pytest.approx(1.0)

    def test_very_low_value_clipped_to_zero(self):
        v = _robust_normalize(-1e9, median=0.0, mad=1.0, clip_sigma=3.0)
        assert v == pytest.approx(0.0)

    def test_output_in_01(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = float(rng.standard_normal(1)[0] * 10)
            v = _robust_normalize(x, median=0.0, mad=2.0, clip_sigma=3.0)
            assert 0.0 <= v <= 1.0

    def test_zero_mad_uses_epsilon(self):
        # MAD=0, epsilon prevents div-by-zero
        v = _robust_normalize(5.0, median=5.0, mad=0.0, epsilon=1e-8, clip_sigma=3.0)
        assert v == pytest.approx(0.5)

    def test_above_median_maps_above_half(self):
        v = _robust_normalize(6.0, median=5.0, mad=1.0, clip_sigma=3.0)
        assert v > 0.5

    def test_below_median_maps_below_half(self):
        v = _robust_normalize(4.0, median=5.0, mad=1.0, clip_sigma=3.0)
        assert v < 0.5


class TestSigmoidNormalize:
    def test_at_center_is_half(self):
        v = _sigmoid_normalize(0.0, center=0.0, scale=1.0)
        assert v == pytest.approx(0.5)

    def test_output_in_01(self):
        for x in np.linspace(-100, 100, 50):
            v = _sigmoid_normalize(float(x), center=0.0, scale=1.0)
            assert 0.0 <= v <= 1.0

    def test_zero_scale_uses_epsilon(self):
        # Should not crash, output should be in [0, 1]
        v = _sigmoid_normalize(1.0, center=0.0, scale=0.0, epsilon=1e-8)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# AdaptiveNormalizer tests
# ---------------------------------------------------------------------------

class TestAdaptiveNormalizer:
    @pytest.fixture
    def config(self):
        return AcousticConfig.development_16khz()

    def _make_stats(self, median: float = 0.1, mad: float = 0.05) -> dict:
        snap = _snap(median, mad)
        return {
            "rms":              snap,
            "zcr":              snap,
            "spectral_flux":    snap,
            "spectral_entropy": snap,
            "centroid_var":     snap,
            "estimated_snr_db": snap,
        }

    def test_all_outputs_in_01(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec()
        stats = self._make_stats()
        nfv = norm.normalize(fv, stats)
        for attr in [
            "rms_norm", "zcr_norm", "spectral_flux_norm",
            "spectral_entropy_norm", "centroid_var_norm",
            "transient_norm", "snr_difficulty_norm"
        ]:
            v = getattr(nfv, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} out of [0,1]"

    def test_zero_zcr_maps_to_zero(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec(zcr=0.0)
        nfv = norm.normalize(fv, self._make_stats())
        assert nfv.zcr_norm == pytest.approx(0.0)

    def test_max_zcr_maps_to_one(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec(zcr=1.0)
        nfv = norm.normalize(fv, self._make_stats())
        assert nfv.zcr_norm == pytest.approx(1.0)

    def test_max_entropy_maps_to_one(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec(entropy=1.0)
        nfv = norm.normalize(fv, self._make_stats())
        assert nfv.spectral_entropy_norm == pytest.approx(1.0)

    def test_zero_transient_maps_to_zero(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec(transient=0.0)
        nfv = norm.normalize(fv, self._make_stats())
        assert nfv.transient_norm == pytest.approx(0.0)

    def test_high_snr_gives_low_difficulty(self, config):
        norm = AdaptiveNormalizer(config)
        # High SNR → high snr_norm → 1 - high = low difficulty
        fv_high = _feature_vec(snr_db=30.0)
        fv_low = _feature_vec(snr_db=-5.0)
        stats_h = {
            "rms": _snap(0.1, 0.05),
            "zcr": _snap(0.3, 0.1),
            "spectral_flux": _snap(0.5, 0.2),
            "spectral_entropy": _snap(0.5, 0.1),
            "centroid_var": _snap(0.1, 0.05),
            "estimated_snr_db": _snap(10.0, 10.0),
        }
        nfv_high = norm.normalize(fv_high, stats_h)
        nfv_low = norm.normalize(fv_low, stats_h)
        assert nfv_high.snr_difficulty_norm < nfv_low.snr_difficulty_norm

    def test_uses_calibration_when_available(self, config):
        cal = NormalizationCalibration(
            rms=FeatureCalibration(median=0.2, mad=0.05),
        )
        norm = AdaptiveNormalizer(config, calibration=cal)
        fv = _feature_vec(rms=0.2)
        stats = self._make_stats()
        nfv = norm.normalize(fv, stats)
        # At median, should map to ~0.5
        assert nfv.rms_norm == pytest.approx(0.5, abs=0.05)

    def test_outlier_flag_propagated(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec()
        stats = self._make_stats()
        stats["rms"] = _snap(0.1, 0.05, is_outlier=True)
        nfv = norm.normalize(fv, stats)
        assert nfv.outlier_flags.get("rms") is True

    def test_constant_feature_handled(self, config):
        norm = AdaptiveNormalizer(config)
        fv = _feature_vec(flux=5.0)
        # All same value → MAD=0, epsilon protects
        stats = self._make_stats(median=5.0, mad=0.0)
        nfv = norm.normalize(fv, stats)
        assert 0.0 <= nfv.spectral_flux_norm <= 1.0


# ---------------------------------------------------------------------------
# NormalizationCalibration tests
# ---------------------------------------------------------------------------

class TestNormalizationCalibration:
    def test_from_dev_features(self):
        # Build synthetic feature vectors
        fvs = []
        for i in range(50):
            fv = _feature_vec(
                rms=0.1 * (i + 1) / 50,
                zcr=0.5,
                flux=float(i),
            )
            fvs.append(fv)
        cal = NormalizationCalibration.from_dev_features(fvs)
        assert cal.rms.is_calibrated
        assert cal.spectral_flux.is_calibrated
        assert cal.rms.median > 0

    def test_empty_features_returns_defaults(self):
        cal = NormalizationCalibration.from_dev_features([])
        assert not cal.rms.is_calibrated

    def test_calibration_applied(self):
        config = AcousticConfig.development_16khz()
        fvs = [_feature_vec(rms=1.0) for _ in range(20)]
        cal = NormalizationCalibration.from_dev_features(fvs)
        norm = AdaptiveNormalizer(config, calibration=cal)
        fv = _feature_vec(rms=1.0)
        from person5_acoustic.adaptive_stats import AdaptiveStatsSnapshot
        stats = {k: AdaptiveStatsSnapshot(0.0, 0.0, 0.0, 0.0, False)
                 for k in ["rms", "zcr", "spectral_flux", "spectral_entropy",
                            "centroid_var", "estimated_snr_db"]}
        nfv = norm.normalize(fv, stats)
        # At calibration median, should be ~0.5
        assert nfv.rms_norm == pytest.approx(0.5, abs=0.1)
