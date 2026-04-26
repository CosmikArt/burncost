"""Tests for :class:`TrendEstimator`."""

from __future__ import annotations

import numpy as np
import pytest

from burncost import TrendEstimator


@pytest.fixture
def years():
    return np.arange(2018, 2026, dtype=float)


@pytest.fixture
def exp_severity(years):
    # y = 1000 * exp(0.05 * (t - 2018)) — clean exponential.
    return 1_000.0 * np.exp(0.05 * (years - 2018))


@pytest.fixture
def linear_severity(years):
    return 1_000.0 + 50.0 * (years - 2018)


class TestFitMethods:
    def test_unknown_method_raises(self, years, linear_severity):
        with pytest.raises(ValueError, match="Unknown trend method"):
            TrendEstimator().fit(years, linear_severity, method="bogus")

    def test_shape_mismatch_raises(self, years):
        with pytest.raises(ValueError, match="same shape"):
            TrendEstimator().fit(years, np.array([1.0, 2.0]), method="linear")

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="At least two"):
            TrendEstimator().fit([2024.0], [1.0], method="linear")

    def test_exponential_requires_positive_y(self, years):
        with pytest.raises(ValueError, match="positive"):
            TrendEstimator().fit(years, np.zeros_like(years), method="exponential")

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            TrendEstimator().predict([2024.0])


class TestExponential:
    def test_exponential_recovers_rate(self, years, exp_severity):
        trend = TrendEstimator().fit(years, exp_severity, method="exponential")
        assert pytest.approx(trend.params["b"], rel=1e-6) == 0.05
        assert pytest.approx(trend.annual_rate(), rel=1e-6) == np.exp(0.05) - 1

    def test_predict_round_trip(self, years, exp_severity):
        trend = TrendEstimator().fit(years, exp_severity, method="exponential")
        np.testing.assert_allclose(trend.predict(years), exp_severity, rtol=1e-8)

    def test_trend_factor_compounds(self, years, exp_severity):
        trend = TrendEstimator().fit(years, exp_severity, method="exponential")
        factor = trend.trend_factor(2022, 2026)
        assert pytest.approx(factor, rel=1e-8) == np.exp(0.05 * 4)


class TestLinear:
    def test_linear_recovers_slope(self, years, linear_severity):
        trend = TrendEstimator().fit(years, linear_severity, method="linear")
        assert pytest.approx(trend.params["b"], rel=1e-8) == 50.0
        assert pytest.approx(trend.predict([2026])[0], rel=1e-8) == 1_000 + 50 * 8

    def test_linear_trend_factor_is_ratio(self, years, linear_severity):
        trend = TrendEstimator().fit(years, linear_severity, method="linear")
        factor = trend.trend_factor(2022, 2024)
        expected = (1_000 + 50 * 6) / (1_000 + 50 * 4)
        assert pytest.approx(factor, rel=1e-8) == expected


class TestOtherMethods:
    def test_log_linear_matches_exponential(self, years, exp_severity):
        a = TrendEstimator().fit(years, exp_severity, method="exponential")
        b = TrendEstimator().fit(years, exp_severity, method="log_linear")
        np.testing.assert_allclose(a.predict(years), b.predict(years), rtol=1e-12)

    def test_multiplicative_matches_exponential(self, years, exp_severity):
        a = TrendEstimator().fit(years, exp_severity, method="exponential")
        b = TrendEstimator().fit(years, exp_severity, method="multiplicative")
        np.testing.assert_allclose(a.predict(years), b.predict(years), rtol=1e-12)

    def test_additive_matches_linear(self, years, linear_severity):
        a = TrendEstimator().fit(years, linear_severity, method="linear")
        b = TrendEstimator().fit(years, linear_severity, method="additive")
        np.testing.assert_allclose(a.predict(years), b.predict(years), rtol=1e-12)

    def test_power_recovers_exponent(self):
        x = np.arange(1, 11, dtype=float)
        y = 3.0 * x ** 1.5
        trend = TrendEstimator().fit(x, y, method="power")
        assert pytest.approx(trend.params["b"], rel=1e-6) == 1.5
        assert pytest.approx(trend.params["a"], rel=1e-6) == 3.0

    def test_power_requires_positive_x(self):
        with pytest.raises(ValueError, match="positive"):
            TrendEstimator().fit([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], method="power")

    def test_mixed_blends_exponential_and_linear(self, years, exp_severity):
        mixed = TrendEstimator().fit(years, exp_severity, method="mixed")
        exp_only = TrendEstimator().fit(years, exp_severity, method="exponential")
        lin_only = TrendEstimator().fit(years, exp_severity, method="linear")
        expected = 0.5 * (exp_only.predict(years) + lin_only.predict(years))
        np.testing.assert_allclose(mixed.predict(years), expected, rtol=1e-12)


class TestPlot:
    def test_plot_returns_axes(self, years, exp_severity):
        import matplotlib

        matplotlib.use("Agg", force=True)
        trend = TrendEstimator().fit(years, exp_severity, method="exponential")
        ax = trend.plot()
        assert ax is not None
        # Should have one scatter (observed) + one line (fit).
        assert len(ax.collections) >= 1
        assert len(ax.lines) >= 1

    def test_plot_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            TrendEstimator().plot()
