"""Tests for burncost.diagnostics."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from burncost import DevelopmentFactors, LossTriangle, TrendEstimator
from burncost.diagnostics import (
    chain_ladder_residuals,
    development_factor_plot,
    development_stability,
    goodness_of_fit,
    trend_fit_summary,
    trend_residual_plot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_exponential() -> TrendEstimator:
    years = np.arange(2018, 2026, dtype=float)
    sev = 1_000.0 * np.exp(0.05 * (years - 2018))
    return TrendEstimator().fit(years, sev, method="exponential")


@pytest.fixture
def fitted_linear() -> TrendEstimator:
    years = np.arange(2018, 2026, dtype=float)
    y = 100.0 + 5.0 * (years - 2018) + np.array(
        [0.0, 0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1]
    )
    return TrendEstimator().fit(years, y, method="linear")


@pytest.fixture
def perfect_triangle() -> LossTriangle:
    """Triangle where every age-to-age ratio is exactly 1.5; residuals zero."""
    f = 1.5
    base = np.array([1_000.0, 1_100.0, 1_200.0, 1_300.0])
    n = len(base)
    data = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n - i):
            data[i, j] = base[i] * (f ** j)
    return LossTriangle(
        data,
        accident_years=[2022, 2023, 2024, 2025],
        dev_periods=[12, 24, 36, 48],
    )


@pytest.fixture
def noisy_triangle() -> LossTriangle:
    data = np.array([
        [1_000.0, 1_500.0, 1_800.0, 1_900.0],
        [1_100.0, 1_650.0, 1_980.0, np.nan],
        [1_200.0, 1_800.0, np.nan, np.nan],
        [1_300.0, np.nan, np.nan, np.nan],
    ])
    return LossTriangle(
        data,
        accident_years=[2022, 2023, 2024, 2025],
        dev_periods=[12, 24, 36, 48],
    )


# ---------------------------------------------------------------------------
# trend_fit_summary
# ---------------------------------------------------------------------------

class TestTrendFitSummary:
    def test_returns_expected_keys(self, fitted_exponential):
        summary = trend_fit_summary(fitted_exponential)
        assert set(summary.keys()) == {
            "method", "n_observations", "r_squared", "rmse", "mae",
            "residual_std",
        }

    def test_clean_exponential_has_high_r_squared(self, fitted_exponential):
        summary = trend_fit_summary(fitted_exponential)
        assert 0.0 <= summary["r_squared"] <= 1.0
        # Synthesised data is deterministic; r^2 should be ~1.
        assert summary["r_squared"] > 0.999

    def test_method_round_trips(self, fitted_linear):
        assert trend_fit_summary(fitted_linear)["method"] == "linear"

    def test_unfit_estimator_raises(self):
        with pytest.raises(RuntimeError):
            trend_fit_summary(TrendEstimator())


# ---------------------------------------------------------------------------
# trend_residual_plot
# ---------------------------------------------------------------------------

class TestTrendResidualPlot:
    def test_returns_axes(self, fitted_exponential):
        ax = trend_residual_plot(fitted_exponential)
        assert isinstance(ax, Axes)

    def test_runs_headless(self, fitted_linear):
        # No exception under matplotlib.use("Agg") at module top.
        trend_residual_plot(fitted_linear)


# ---------------------------------------------------------------------------
# development_stability
# ---------------------------------------------------------------------------

class TestDevelopmentStability:
    def test_returns_dataframe_with_expected_shape(self, noisy_triangle):
        dev = DevelopmentFactors.from_triangle(noisy_triangle)
        stab = development_stability(dev)
        assert isinstance(stab, pd.DataFrame)
        assert list(stab.columns) == [
            "mean", "median", "std", "min", "max", "cv",
        ]
        # Three intervals: 12-24, 24-36, 36-48.
        assert list(stab.index) == ["12-24", "24-36", "36-48"]

    def test_cv_non_negative_or_nan(self, noisy_triangle):
        dev = DevelopmentFactors.from_triangle(noisy_triangle)
        stab = development_stability(dev)
        cvs = stab["cv"].dropna().values
        assert np.all(cvs >= 0)


# ---------------------------------------------------------------------------
# development_factor_plot
# ---------------------------------------------------------------------------

class TestDevelopmentFactorPlot:
    def test_returns_axes(self, noisy_triangle):
        dev = DevelopmentFactors.from_triangle(noisy_triangle)
        ax = development_factor_plot(dev)
        assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# chain_ladder_residuals
# ---------------------------------------------------------------------------

class TestChainLadderResiduals:
    def test_zero_residuals_on_deterministic_triangle(self, perfect_triangle):
        dev = DevelopmentFactors.from_triangle(perfect_triangle)
        resid = chain_ladder_residuals(dev)
        observable = resid.values[~np.isnan(resid.values)]
        # On a perfectly deterministic triangle, residuals must be 0 (within fp tol).
        assert np.allclose(observable, 0.0, atol=1e-10)

    def test_shape_matches_intervals(self, noisy_triangle):
        dev = DevelopmentFactors.from_triangle(noisy_triangle)
        resid = chain_ladder_residuals(dev)
        assert list(resid.columns) == ["12-24", "24-36", "36-48"]
        assert list(resid.index) == [2022, 2023, 2024, 2025]

    def test_unobservable_cells_are_nan(self, noisy_triangle):
        dev = DevelopmentFactors.from_triangle(noisy_triangle)
        resid = chain_ladder_residuals(dev)
        # AY 2025 has only the 12-month value, so all its residuals are NaN.
        assert resid.loc[2025].isna().all()


# ---------------------------------------------------------------------------
# goodness_of_fit
# ---------------------------------------------------------------------------

class TestGoodnessOfFit:
    def test_returns_floats(self, fitted_exponential):
        gof = goodness_of_fit(fitted_exponential)
        assert set(gof.keys()) == {"loglik", "aic", "bic", "r_squared"}
        for v in gof.values():
            assert isinstance(v, float)

    def test_bic_at_least_aic_for_reasonable_n(self, fitted_linear):
        # For n >= 8 (here n=8), log(n) >= 2, so BIC penalty >= AIC penalty.
        gof = goodness_of_fit(fitted_linear)
        assert gof["bic"] >= gof["aic"]

    def test_r_squared_in_unit_interval_for_clean_fit(self, fitted_exponential):
        gof = goodness_of_fit(fitted_exponential)
        assert 0.0 <= gof["r_squared"] <= 1.0

    def test_zero_rss_yields_infinite_loglik(self):
        # Force an exactly-zero residual sum of squares by setting the fit
        # parameters so predict() reproduces y to the bit. polyfit() leaves
        # tiny FP noise so we override the params directly.
        years = np.array([2020.0, 2021.0, 2022.0])
        y = np.array([100.0, 100.0, 100.0])
        est = TrendEstimator()
        est._method = "additive"
        est._x = years
        est._y = y
        est._params = {"a": 100.0, "b": 0.0}
        gof = goodness_of_fit(est)
        assert math.isinf(gof["loglik"])


# ---------------------------------------------------------------------------
# Defensive branches
# ---------------------------------------------------------------------------

class TestDefensiveBranches:
    def test_development_stability_no_valid_pairs_in_interval(self):
        # Triangle where the 12-24 column has no aligned pairs (column 0 is
        # all NaN), so the interval has zero observations -> all-NaN row.
        data = np.array([
            [np.nan, 1500.0, 1800.0],
            [np.nan, 1650.0, np.nan],
            [np.nan, np.nan, np.nan],
        ])
        tri = LossTriangle(
            data,
            accident_years=[2022, 2023, 2024],
            dev_periods=[12, 24, 36],
        )
        dev = DevelopmentFactors.from_triangle(tri)
        stab = development_stability(dev)
        assert stab.loc["12-24"].isna().all()

    def test_chain_ladder_residuals_skips_when_factor_is_nan(self):
        # Same triangle as above: selected factor for 12-24 is NaN, so the
        # corresponding residual column is entirely NaN.
        data = np.array([
            [np.nan, 1500.0, 1800.0],
            [np.nan, 1650.0, np.nan],
            [np.nan, np.nan, np.nan],
        ])
        tri = LossTriangle(
            data,
            accident_years=[2022, 2023, 2024],
            dev_periods=[12, 24, 36],
        )
        dev = DevelopmentFactors.from_triangle(tri)
        resid = chain_ladder_residuals(dev)
        assert resid["12-24"].isna().all()
