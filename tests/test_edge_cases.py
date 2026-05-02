"""Edge-case tests targeting the remaining branches of :mod:`burncost.core`.

These tests exist to drive line coverage to 100%; they exercise validation
paths, defensive branches, and the lesser-used permutations of the public
API (e.g. ``annual_rate`` for every trend method, ``power`` predict,
single-column triangle, all-NaN AY row, exponential tail edge cases).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from burncost import (
    DevelopmentFactors,
    LossTriangle,
    OnLevelPremium,
    TrendEstimator,
)


# ---------------------------------------------------------------------------
# LossTriangle
# ---------------------------------------------------------------------------

def test_to_incremental_on_already_incremental_returns_self():
    inc = np.array([[100.0, 50.0], [110.0, np.nan]])
    tri = LossTriangle(inc, [2024, 2025], [12, 24], cumulative=False)
    assert tri.to_incremental() is tri


# ---------------------------------------------------------------------------
# TrendEstimator
# ---------------------------------------------------------------------------

class TestTrendEdgeCases:
    def test_method_property_before_fit(self):
        assert TrendEstimator().method is None

    def test_params_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            _ = TrendEstimator().params

    def test_trend_factor_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            TrendEstimator().trend_factor(2022, 2024)

    def test_annual_rate_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            TrendEstimator().annual_rate()

    def test_fit_rejects_non_finite_input(self):
        with pytest.raises(ValueError, match="finite"):
            TrendEstimator().fit(
                np.array([2020.0, 2021.0, 2022.0]),
                np.array([1.0, np.nan, 3.0]),
                method="linear",
            )

    def test_mixed_requires_positive_y(self):
        with pytest.raises(ValueError, match="positive"):
            TrendEstimator().fit(
                np.array([2020.0, 2021.0, 2022.0]),
                np.array([1.0, -1.0, 2.0]),
                method="mixed",
            )

    def test_power_predict(self):
        x = np.arange(1, 11, dtype=float)
        y = 2.0 * x ** 0.5
        trend = TrendEstimator().fit(x, y, method="power")
        np.testing.assert_allclose(trend.predict(x), y, rtol=1e-8)

    def test_trend_factor_zero_base_raises(self):
        # Bypass fit and inject params giving an exact 0 at the base year.
        trend = TrendEstimator()
        trend._method = "linear"
        trend._params = {"a": 0.0, "b": 0.0}
        trend._x = np.array([2020.0, 2021.0])
        trend._y = np.array([0.0, 0.0])
        with pytest.raises(ZeroDivisionError):
            trend.trend_factor(2022, 2024)

    def test_annual_rate_linear(self):
        x = np.array([2020.0, 2021.0, 2022.0])
        y = np.array([100.0, 105.0, 110.0])  # mean 105, slope 5 => rate 5/105
        trend = TrendEstimator().fit(x, y, method="linear")
        assert pytest.approx(trend.annual_rate()) == 5.0 / 105.0

    def test_annual_rate_linear_zero_mean(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([-1.0, 0.0, 1.0])  # mean 0
        trend = TrendEstimator().fit(x, y, method="linear")
        assert trend.annual_rate() == 0.0

    def test_annual_rate_power(self):
        x = np.arange(1, 6, dtype=float)
        y = 2.0 * x ** 0.5
        trend = TrendEstimator().fit(x, y, method="power")
        # b / mean(x); mean(x) = 3, b = 0.5 => 1/6
        assert pytest.approx(trend.annual_rate()) == 0.5 / 3.0

    def test_annual_rate_mixed(self):
        years = np.arange(2018, 2026, dtype=float)
        sev = 1_000.0 * np.exp(0.05 * (years - 2018))
        trend = TrendEstimator().fit(years, sev, method="mixed")
        # Just verify it returns something finite. Exact value follows formula.
        assert np.isfinite(trend.annual_rate())

    def test_annual_rate_mixed_zero_mean_y(self):
        # Force the additive component to fall back to 0 when mean(y) == 0.
        # We can't fit through "mixed" (it requires y > 0), so inject params.
        trend = TrendEstimator()
        trend._method = "mixed"
        trend._params = {"a_exp": 1.0, "b_exp": 0.0, "a_lin": 0.0, "b_lin": 1.0}
        trend._x = np.array([0.0, 1.0])
        trend._y = np.array([0.0, 0.0])  # mean = 0 triggers fallback
        rate = trend.annual_rate()
        # exp(0) - 1 = 0 + linear fallback 0 -> 0.5 * 0 = 0
        assert rate == 0.0

    def test_plot_with_provided_axes(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        years = np.arange(2018, 2026, dtype=float)
        sev = 1_000.0 * np.exp(0.05 * (years - 2018))
        trend = TrendEstimator().fit(years, sev, method="exponential")
        _, ax = plt.subplots()
        returned = trend.plot(ax=ax)
        assert returned is ax


# ---------------------------------------------------------------------------
# DevelopmentFactors
# ---------------------------------------------------------------------------

class TestDevelopmentEdgeCases:
    def test_link_ratio_returns_nan_when_no_valid_pairs(self):
        # Two columns, but column 0 is all NaN, so no valid (a, b) pairs.
        data = np.array(
            [
                [np.nan, 100.0],
                [np.nan, 110.0],
            ]
        )
        tri = LossTriangle(data, [2024, 2025], [12, 24])
        dev = DevelopmentFactors.from_triangle(tri)
        sel = dev.selected_factors()
        assert np.isnan(sel.iloc[0])

    def test_age_to_age_with_single_column(self):
        data = np.array([[100.0], [110.0]])
        tri = LossTriangle(data, [2024, 2025], [12])
        dev = DevelopmentFactors.from_triangle(tri)
        df = dev.age_to_age()
        assert df.empty
        assert df.shape == (2, 0)
        assert list(df.index) == [2024, 2025]

    def test_tail_factor_exponential_insufficient_factors(self):
        # Selected factors that are mostly <= 1 (no growth left to extrapolate).
        data = np.array(
            [
                [100.0, 99.0, 98.5, 98.5],
                [100.0, 99.0, 98.5, np.nan],
                [100.0, 99.0, np.nan, np.nan],
                [100.0, np.nan, np.nan, np.nan],
            ]
        )
        tri = LossTriangle(data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        dev = DevelopmentFactors.from_triangle(tri)
        assert dev.tail_factor("exponential") == 1.0

    def test_tail_factor_exponential_slow_decay_runs_full_loop(self):
        # Selected factors with very slow decay so abs(f_k - 1) never falls
        # below 1e-6 within 50 iterations, exercising the loop's natural exit.
        # Use roughly constant excess factors slightly above 1.
        data = np.array(
            [
                [100.0, 102.0, 104.04, 106.12],
                [100.0, 102.0, 104.04, np.nan],
                [100.0, 102.0, np.nan, np.nan],
                [100.0, np.nan, np.nan, np.nan],
            ]
        )
        tri = LossTriangle(data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        dev = DevelopmentFactors.from_triangle(tri)
        tail = dev.tail_factor("exponential")
        # Decay refused (slope ~ 0) -> 1.0; or accepted with slow decay > 1.0.
        assert tail >= 1.0

    def test_tail_factor_exponential_non_decaying(self):
        # Selected factors that grow over time (slope >= 0); should refuse.
        data = np.array(
            [
                [100.0, 110.0, 130.0, 200.0],
                [100.0, 110.0, 130.0, np.nan],
                [100.0, 110.0, np.nan, np.nan],
                [100.0, np.nan, np.nan, np.nan],
            ]
        )
        tri = LossTriangle(data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        dev = DevelopmentFactors.from_triangle(tri)
        assert dev.tail_factor("exponential") == 1.0

    def test_ultimate_losses_skips_all_nan_row(self):
        data = np.array(
            [
                [1_000.0, 1_500.0, 1_800.0, 1_900.0],
                [1_100.0, 1_650.0, 1_980.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],  # AY with no observations
                [1_300.0, np.nan, np.nan, np.nan],
            ]
        )
        tri = LossTriangle(data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        dev = DevelopmentFactors.from_triangle(tri)
        ult = dev.ultimate_losses()
        assert np.isnan(ult.loc[2024])
        assert not np.isnan(ult.loc[2025])


# ---------------------------------------------------------------------------
# OnLevelPremium
# ---------------------------------------------------------------------------

class TestOnLevelEdgeCases:
    def test_apply_rate_changes_rejects_nan_change(self):
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2023-01-01"]),
                "change": [np.nan],
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            OnLevelPremium().apply_rate_changes(df)

    def test_apply_rate_changes_rejects_negative_one(self):
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2023-01-01"]),
                "change": [-1.0],
            }
        )
        with pytest.raises(ValueError, match="-1"):
            OnLevelPremium().apply_rate_changes(df)

    def test_rate_changes_property_is_none_until_applied(self):
        olp = OnLevelPremium()
        assert olp.rate_changes is None
        assert olp.current_level == 1.0
