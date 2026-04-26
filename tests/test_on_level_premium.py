"""Tests for :class:`OnLevelPremium`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from burncost import OnLevelPremium


@pytest.fixture
def rate_changes():
    return pd.DataFrame(
        {
            "effective_date": pd.to_datetime(
                ["2021-07-01", "2023-01-01", "2024-04-01"]
            ),
            "change": [0.05, 0.08, -0.03],
        }
    )


class TestApplyRateChanges:
    def test_requires_dataframe(self):
        with pytest.raises(TypeError):
            OnLevelPremium().apply_rate_changes({"effective_date": [], "change": []})

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing"):
            OnLevelPremium().apply_rate_changes(pd.DataFrame({"date": []}))

    def test_unsorted_input_is_sorted(self):
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2024-01-01", "2022-01-01"]),
                "change": [0.10, 0.05],
            }
        )
        olp = OnLevelPremium().apply_rate_changes(df)
        rc = olp.rate_changes
        assert rc.iloc[0]["effective_date"] < rc.iloc[1]["effective_date"]

    def test_cumulative_factor_is_compound(self, rate_changes):
        olp = OnLevelPremium().apply_rate_changes(rate_changes)
        rc = olp.rate_changes
        np.testing.assert_allclose(
            rc["cum_factor"].values,
            np.array([1.05, 1.05 * 1.08, 1.05 * 1.08 * 0.97]),
        )

    def test_current_level(self, rate_changes):
        olp = OnLevelPremium().apply_rate_changes(rate_changes)
        assert pytest.approx(olp.current_level) == 1.05 * 1.08 * 0.97


class TestParallelogram:
    def test_no_rate_changes_returns_one(self):
        assert OnLevelPremium().on_level_factor(2024) == 1.0

    def test_change_long_before_year_returns_one(self):
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2010-01-01"]),
                "change": [0.10],
            }
        )
        olp = OnLevelPremium().apply_rate_changes(df)
        # Single change far in the past, so AY is fully at the (current) post-change level.
        assert pytest.approx(olp.on_level_factor(2024)) == 1.0

    def test_change_long_after_year_equals_current_level(self):
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2030-01-01"]),
                "change": [0.10],
            }
        )
        olp = OnLevelPremium().apply_rate_changes(df)
        assert pytest.approx(olp.on_level_factor(2024)) == 1.10

    def test_change_at_jan_1_of_year(self):
        # Rate change exactly on Jan 1 of AY: half the AY at old, half at new
        # under the standard parallelogram calculation with 1-year term.
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2024-01-01"]),
                "change": [0.10],
            }
        )
        olp = OnLevelPremium().apply_rate_changes(df)
        # New-rate area for an annual policy with change at p=0:
        # (1-p)^2 / 2 = 0.5 -> 50% new, 50% old.
        # Average level = 0.5 * 1.0 + 0.5 * 1.1 = 1.05.
        # OLF = 1.10 / 1.05.
        assert pytest.approx(olp.on_level_factor(2024), rel=1e-6) == 1.10 / 1.05

    def test_change_mid_year(self):
        # Change on July 1 of AY (p=0.5): new fraction = (1-0.5)^2 / 2 = 0.125.
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2024-07-01"]),
                "change": [0.10],
            }
        )
        olp = OnLevelPremium().apply_rate_changes(df)
        # Decimal year for 2024-07-01 is 2024 + 182/366 (leap year).
        p = 182 / 366
        new_fraction = (1.0 - p) ** 2 / 2.0
        old_fraction = 1.0 - new_fraction
        avg_level = old_fraction * 1.0 + new_fraction * 1.10
        expected_olf = 1.10 / avg_level
        assert pytest.approx(olp.on_level_factor(2024), rel=1e-6) == expected_olf

    def test_six_month_term(self):
        # With a 6-month term and a change a year before AY, the AY should be
        # entirely at the new (current) level — OLF = 1.0.
        df = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2023-01-01"]),
                "change": [0.10],
            }
        )
        olp = OnLevelPremium().apply_rate_changes(df)
        olf = olp.parallelogram(2024, term_months=6)
        assert pytest.approx(olf) == 1.0

    def test_invalid_term_months(self, rate_changes):
        olp = OnLevelPremium().apply_rate_changes(rate_changes)
        with pytest.raises(ValueError, match="term_months"):
            olp.parallelogram(2024, term_months=0)


class TestVectorisedFactors:
    def test_on_level_factors_returns_series(self, rate_changes):
        olp = OnLevelPremium().apply_rate_changes(rate_changes)
        s = olp.on_level_factors([2022, 2023, 2024])
        assert isinstance(s, pd.Series)
        assert list(s.index) == [2022, 2023, 2024]
        # Most recent year should be closest to (current_level / current_level).
        assert s.loc[2024] >= 0.95 and s.loc[2024] <= 1.05
