"""Tests for :class:`DevelopmentFactors`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from burncost import DevelopmentFactors, LossTriangle


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
def dev(triangle):
    return DevelopmentFactors.from_triangle(triangle)


class TestAgeToAge:
    def test_returns_one_row_per_ay_plus_selected(self, dev):
        df = dev.age_to_age()
        assert "selected" in df.index
        assert len(df) == 5  # 4 AYs + selected
        assert list(df.columns) == ["12-24", "24-36", "36-48"]

    def test_individual_ratios(self, dev):
        df = dev.age_to_age()
        assert pytest.approx(df.loc[2022, "12-24"]) == 1.5
        assert pytest.approx(df.loc[2022, "24-36"]) == 1800 / 1500
        assert pytest.approx(df.loc[2023, "12-24"]) == 1650 / 1100
        # Cells without two consecutive observations are NaN.
        assert pd.isna(df.loc[2025, "12-24"])
        assert pd.isna(df.loc[2024, "24-36"])

    def test_volume_weighted_selected(self, dev):
        df = dev.age_to_age(average="volume")
        # 12-24 weighted: (1500 + 1650 + 1800) / (1000 + 1100 + 1200) = 4950/3300
        assert pytest.approx(df.loc["selected", "12-24"]) == 4950 / 3300

    def test_simple_average_selected(self, dev):
        df = dev.age_to_age(average="simple")
        # 12-24 ratios: 1.5, 1.5, 1.5 => mean 1.5
        assert pytest.approx(df.loc["selected", "12-24"]) == 1.5

    def test_medial_average_drops_extremes(self):
        data = np.array(
            [
                [100.0, 110.0, 120.0],
                [100.0, 130.0, 140.0],
                [100.0, 200.0, np.nan],
                [100.0, np.nan, np.nan],
            ]
        )
        tri = LossTriangle(data, [2021, 2022, 2023, 2024], [12, 24, 36])
        dev = DevelopmentFactors.from_triangle(tri)
        df = dev.age_to_age(average="medial")
        # 12-24 ratios: 1.1, 1.3, 2.0; drop high (2.0) and low (1.1) => 1.3
        assert pytest.approx(df.loc["selected", "12-24"]) == 1.3


class TestSelectedFactors:
    def test_volume_method(self, dev):
        sel = dev.selected_factors("volume")
        assert isinstance(sel, pd.Series)
        assert pytest.approx(sel["12-24"]) == 4950 / 3300
        assert pytest.approx(sel["24-36"]) == (1800 + 1980) / (1500 + 1650)
        assert pytest.approx(sel["36-48"]) == 1900 / 1800

    def test_unknown_method_raises(self, dev):
        with pytest.raises(ValueError, match="averaging method"):
            dev.selected_factors("median")  # type: ignore[arg-type]


class TestAgeToUltimate:
    def test_atu_compounds_selected(self, dev):
        atu = dev.age_to_ultimate()
        sel = dev.selected_factors()
        # last period -> tail (default 1.0)
        assert pytest.approx(atu.iloc[-1]) == 1.0
        assert pytest.approx(atu.iloc[-2]) == sel.iloc[-1]
        assert pytest.approx(atu.iloc[0]) == sel.iloc[0] * sel.iloc[1] * sel.iloc[2]

    def test_atu_with_tail(self, dev):
        atu = dev.age_to_ultimate(tail=1.05)
        assert pytest.approx(atu.iloc[-1]) == 1.05


class TestTailFactor:
    def test_unity(self, dev):
        assert dev.tail_factor("unity") == 1.0
        assert dev.tail_factor("none") == 1.0

    def test_unknown_method(self, dev):
        with pytest.raises(ValueError):
            dev.tail_factor("bogus")

    def test_exponential_decay_returns_value_above_unity(self):
        # Crafted decreasing excess factors so the exponential extrapolation works.
        data = np.array(
            [
                [100.0, 130.0, 145.0, 152.0, 155.5],
                [100.0, 130.0, 145.0, 152.0, np.nan],
                [100.0, 130.0, 145.0, np.nan, np.nan],
                [100.0, 130.0, np.nan, np.nan, np.nan],
                [100.0, np.nan, np.nan, np.nan, np.nan],
            ]
        )
        tri = LossTriangle(data, [2021, 2022, 2023, 2024, 2025], [12, 24, 36, 48, 60])
        dev = DevelopmentFactors.from_triangle(tri)
        tail = dev.tail_factor("exponential")
        assert tail >= 1.0


class TestUltimateLosses:
    def test_chain_ladder(self, dev):
        ultimates = dev.ultimate_losses(method="chain_ladder")
        atu = dev.age_to_ultimate()
        # AY 2022 is fully developed (4 cells) -> ultimate = latest * atu[48] = latest * 1.0
        assert pytest.approx(ultimates.loc[2022]) == 1_900.0
        # AY 2025 only has 12 months: ultimate = 1300 * atu[12]
        assert pytest.approx(ultimates.loc[2025]) == 1_300.0 * atu.iloc[0]

    def test_bf_requires_a_priori(self, dev):
        with pytest.raises(ValueError, match="a_priori"):
            dev.ultimate_losses(method="bornhuetter_ferguson")

    def test_bf_a_priori_length_mismatch(self, dev):
        with pytest.raises(ValueError, match="one value per accident year"):
            dev.ultimate_losses(method="bornhuetter_ferguson", a_priori=[1.0, 2.0])

    def test_bf_formula(self, dev):
        a_priori = np.array([2_000.0, 2_000.0, 2_000.0, 2_000.0])
        ultimates = dev.ultimate_losses(
            method="bornhuetter_ferguson", a_priori=a_priori
        )
        atu = dev.age_to_ultimate()
        # AY 2025: latest 1300 at age 12, atu[12]
        expected = 1_300.0 + 2_000.0 * (1.0 - 1.0 / atu.iloc[0])
        assert pytest.approx(ultimates.loc[2025]) == expected

    def test_unknown_method(self, dev):
        with pytest.raises(ValueError, match="Unknown ultimate-loss method"):
            dev.ultimate_losses(method="hocus_pocus")  # type: ignore[arg-type]
