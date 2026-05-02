"""Regression tests for TrendEstimator numerical robustness fixes (v0.1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from burncost import TrendEstimator


def test_trend_factor_raises_on_near_zero_base():
    """y values that produce near-zero prediction at from_year should
    raise rather than return an astronomical factor."""
    te = TrendEstimator()
    te.fit(np.array([2018, 2019, 2020]),
           np.array([-100.0, -50.0, 0.0]),
           method="linear")
    with pytest.raises(ZeroDivisionError):
        te.trend_factor(2020, 2025)


def test_polyfit_centering_preserves_typical_results():
    """Centering x around its mean should not change typical-case
    trend factors observably."""
    te = TrendEstimator()
    te.fit(np.array([2018, 2019, 2020, 2021, 2022]),
           np.array([1000.0, 1050.0, 1102.0, 1158.0, 1216.0]),
           method="exponential")
    # Typical exponential ~5%/yr -> factor 2022->2025 ~ 1.16
    factor = te.trend_factor(2022, 2025)
    assert 1.10 < factor < 1.20
