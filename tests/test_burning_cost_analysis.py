"""Tests for the end-to-end :class:`BurningCostAnalysis` pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from burncost import (
    BurningCostAnalysis,
    DevelopmentFactors,
    LossTriangle,
    OnLevelPremium,
    TrendEstimator,
)


@pytest.fixture
def triangle():
    data = np.array(
        [
            [1_000.0, 1_500.0, 1_800.0, 1_900.0],
            [1_100.0, 1_650.0, 1_980.0, np.nan],
            [1_200.0, 1_800.0, np.nan, np.nan],
            [1_300.0, np.nan, np.nan, np.nan],
        ]
    )
    return LossTriangle(data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])


@pytest.fixture
def fitted_trend():
    years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], dtype=float)
    severity = 1_000.0 * np.exp(0.05 * (years - 2018))
    trend = TrendEstimator()
    trend.fit(years, severity, method="exponential")
    return trend


@pytest.fixture
def on_level():
    rate_changes = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(
                ["2021-07-01", "2023-01-01", "2024-04-01"]
            ),
            "change": [0.05, 0.08, -0.03],
        }
    )
    olp = OnLevelPremium()
    olp.apply_rate_changes(rate_changes)
    return olp


class TestRun:
    def test_run_columns_without_premium(self, triangle, fitted_trend, on_level):
        bc = BurningCostAnalysis(triangle, fitted_trend, on_level)
        df = bc.run(target_year=2026)
        assert set(["ultimate_loss", "trend_factor", "trended_loss", "on_level_factor"]) <= set(df.columns)
        assert "loss_ratio" not in df.columns

    def test_run_with_premium_adds_loss_ratio(
        self, triangle, fitted_trend, on_level
    ):
        premium = {2022: 4_000.0, 2023: 4_200.0, 2024: 4_400.0, 2025: 4_600.0}
        bc = BurningCostAnalysis(
            triangle, fitted_trend, on_level, earned_premium=premium
        )
        df = bc.run(target_year=2026)
        for col in ("earned_premium", "on_level_premium", "loss_ratio", "burning_cost"):
            assert col in df.columns
        # Loss ratio = trended / on-leveled premium
        for ay in df.index:
            expected = df.loc[ay, "trended_loss"] / df.loc[ay, "on_level_premium"]
            assert pytest.approx(df.loc[ay, "loss_ratio"]) == expected

    def test_trend_factor_matches_estimator(self, triangle, fitted_trend, on_level):
        bc = BurningCostAnalysis(triangle, fitted_trend, on_level)
        df = bc.run(target_year=2026)
        for ay in df.index:
            expected = fitted_trend.trend_factor(ay, 2026)
            assert pytest.approx(df.loc[ay, "trend_factor"]) == expected

    def test_ultimate_matches_dev_factors(self, triangle, fitted_trend, on_level):
        bc = BurningCostAnalysis(triangle, fitted_trend, on_level)
        df = bc.run(target_year=2026)
        ult_direct = DevelopmentFactors.from_triangle(triangle).ultimate_losses()
        for ay in df.index:
            assert pytest.approx(df.loc[ay, "ultimate_loss"]) == ult_direct.loc[ay]

    def test_summary_requires_run(self, triangle, fitted_trend, on_level):
        bc = BurningCostAnalysis(triangle, fitted_trend, on_level)
        with pytest.raises(RuntimeError, match="run"):
            bc.summary()

    def test_summary_returns_last_run(self, triangle, fitted_trend, on_level):
        bc = BurningCostAnalysis(triangle, fitted_trend, on_level)
        df = bc.run(target_year=2026)
        summary = bc.summary()
        pd.testing.assert_frame_equal(df, summary)
        # Mutating the summary copy must not affect the stored result.
        summary.iloc[0, 0] = -1
        assert bc.summary().iloc[0, 0] != -1

    def test_run_with_bf_method(self, triangle, fitted_trend, on_level):
        bc = BurningCostAnalysis(triangle, fitted_trend, on_level)
        a_priori = np.array([2_000.0, 2_100.0, 2_200.0, 2_300.0])
        df = bc.run(
            target_year=2026,
            development_method="bornhuetter_ferguson",
            a_priori=a_priori,
        )
        assert df["ultimate_loss"].notna().all()
