"""Tests for the :class:`LossTriangle` container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from burncost import LossTriangle


@pytest.fixture
def cumulative_data():
    return np.array(
        [
            [1_000.0, 1_500.0, 1_800.0, 1_900.0],
            [1_100.0, 1_650.0, 1_980.0, np.nan],
            [1_200.0, 1_800.0, np.nan, np.nan],
            [1_300.0, np.nan, np.nan, np.nan],
        ]
    )


@pytest.fixture
def incremental_data():
    # Same triangle as `cumulative_data` but expressed incrementally.
    return np.array(
        [
            [1_000.0, 500.0, 300.0, 100.0],
            [1_100.0, 550.0, 330.0, np.nan],
            [1_200.0, 600.0, np.nan, np.nan],
            [1_300.0, np.nan, np.nan, np.nan],
        ]
    )


class TestConstruction:
    def test_basic_attributes(self, cumulative_data):
        tri = LossTriangle(
            cumulative_data,
            accident_years=[2022, 2023, 2024, 2025],
            dev_periods=[12, 24, 36, 48],
        )
        assert tri.shape == (4, 4)
        assert tri.is_cumulative is True
        assert tri.accident_years == [2022, 2023, 2024, 2025]
        assert tri.dev_periods == [12, 24, 36, 48]

    def test_repr_includes_metadata(self, cumulative_data):
        tri = LossTriangle(cumulative_data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        text = repr(tri)
        assert "LossTriangle" in text
        assert "2022" in text and "48" in text

    @pytest.mark.parametrize(
        "data, ays, dps",
        [
            (np.array([1.0, 2.0, 3.0]), [2023], [12, 24, 36]),  # 1-D
            (np.zeros((2, 3)), [2023], [12, 24, 36]),  # row mismatch
            (np.zeros((2, 3)), [2023, 2024], [12, 24]),  # col mismatch
        ],
    )
    def test_shape_validation(self, data, ays, dps):
        with pytest.raises(ValueError):
            LossTriangle(data, ays, dps)


class TestConversions:
    def test_to_cumulative_idempotent(self, cumulative_data):
        tri = LossTriangle(cumulative_data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        assert tri.to_cumulative() is tri

    def test_incremental_to_cumulative(self, incremental_data, cumulative_data):
        tri = LossTriangle(
            incremental_data,
            [2022, 2023, 2024, 2025],
            [12, 24, 36, 48],
            cumulative=False,
        )
        cum = tri.to_cumulative()
        assert cum.is_cumulative
        np.testing.assert_allclose(
            cum.values, cumulative_data, equal_nan=True
        )

    def test_cumulative_to_incremental(self, cumulative_data, incremental_data):
        tri = LossTriangle(cumulative_data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        inc = tri.to_incremental()
        assert inc.is_cumulative is False
        np.testing.assert_allclose(
            inc.values, incremental_data, equal_nan=True
        )

    def test_round_trip_preserves_values(self, incremental_data):
        tri = LossTriangle(
            incremental_data,
            [2022, 2023, 2024, 2025],
            [12, 24, 36, 48],
            cumulative=False,
        )
        round_tripped = tri.to_cumulative().to_incremental()
        np.testing.assert_allclose(
            round_tripped.values, incremental_data, equal_nan=True
        )

    def test_to_dataframe(self, cumulative_data):
        tri = LossTriangle(cumulative_data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        df = tri.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 4)
        assert df.index.name == "accident_year"
        assert df.columns.name == "dev_period"
        assert df.loc[2022, 12] == 1_000.0
        assert pd.isna(df.loc[2025, 24])


class TestLatestDiagonal:
    def test_latest_diagonal_picks_last_observed(self, cumulative_data):
        tri = LossTriangle(cumulative_data, [2022, 2023, 2024, 2025], [12, 24, 36, 48])
        latest = tri.latest_diagonal
        assert latest.loc[2022] == 1_900.0
        assert latest.loc[2023] == 1_980.0
        assert latest.loc[2024] == 1_800.0
        assert latest.loc[2025] == 1_300.0

    def test_latest_diagonal_handles_all_nan_row(self):
        data = np.array([[1.0, 2.0], [np.nan, np.nan]])
        tri = LossTriangle(data, [2024, 2025], [12, 24])
        latest = tri.latest_diagonal
        assert latest.loc[2024] == 2.0
        assert pd.isna(latest.loc[2025])
