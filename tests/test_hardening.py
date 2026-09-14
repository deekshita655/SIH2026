"""Regression tests for the reference-quality hardening utility."""
from __future__ import annotations

import numpy as np
import pytest

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


def test_disabled_quality_controller_keeps_full_adaptation():
    result = estimate_reference_quality(np.zeros(16), np.zeros(16), ReferenceQualityConfig(enabled=False))
    assert result.state == "GOOD"
    assert result.mu_scale == pytest.approx(1.0)
