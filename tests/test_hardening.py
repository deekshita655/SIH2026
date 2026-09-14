"""Regression tests for integration hardening."""
from __future__ import annotations

import numpy as np
import pytest

from integration.config import NLMSConfig
from integration.nlms import NLMSFilter
from integration.reference_quality import ReferenceQualityConfig, estimate_reference_quality


def test_correlated_reference_is_good():
    rng = np.random.default_rng(7)
    reference = rng.standard_normal(2048)
    primary = 0.8 * reference + 0.05 * rng.standard_normal(2048)
    result = estimate_reference_quality(primary, reference)
    assert result.state == "GOOD"
    assert result.score >= 0.65
    assert result.mu_scale == pytest.approx(1.0)


def test_uncorrelated_reference_is_poor():
    rng = np.random.default_rng(8)
    result = estimate_reference_quality(rng.standard_normal(2048), rng.standard_normal(2048))
    assert result.state == "POOR"
    assert result.mu_scale == pytest.approx(0.0)


def test_empty_reference_is_poor():
    result = estimate_reference_quality(np.ones(10), np.array([]))
    assert result.state == "POOR"
    assert result.mu_scale == pytest.approx(0.0)


def test_disabled_quality_controller_keeps_full_adaptation():
    result = estimate_reference_quality(np.zeros(16), np.zeros(16), ReferenceQualityConfig(enabled=False))
    assert result.state == "GOOD"
    assert result.mu_scale == pytest.approx(1.0)


def test_poor_reference_freezes_nlms_adaptation():
    rng = np.random.default_rng(12)
    primary = rng.standard_normal(4096)
    reference = rng.standard_normal(4096)
    cfg = NLMSConfig(filter_length=16, step_size=0.5, block_size=512, sample_rate=16000)
    result = NLMSFilter(cfg).process(primary, reference, 16000)
    assert result.reference_quality is not None
    assert result.reference_quality.mu_scale == pytest.approx(0.0)
    assert np.allclose(result.estimated_noise, 0.0)
    assert np.allclose(result.cleaned_speech, primary.astype(np.float32))


def test_good_reference_allows_nlms_adaptation():
    rng = np.random.default_rng(13)
    reference = rng.standard_normal(4096)
    primary = 0.7 * reference + 0.02 * rng.standard_normal(4096)
    cfg = NLMSConfig(filter_length=16, step_size=0.5, block_size=512, sample_rate=16000)
    result = NLMSFilter(cfg).process(primary, reference, 16000)
    assert result.reference_quality is not None
    assert result.reference_quality.mu_scale == pytest.approx(1.0)
    assert np.linalg.norm(result.estimated_noise) > 0.0
