"""End-to-end burning cost pipeline.

Wires :class:`LossTriangle` development, :class:`TrendEstimator` projection,
and :class:`OnLevelPremium` adjustment into a single reproducible analysis
object that produces a per-accident-year output table.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from burncost.development import DevelopmentFactors, _AVERAGE_METHODS
from burncost.onlevel import OnLevelPremium
from burncost.trending import TrendEstimator
from burncost.triangle import LossTriangle


class BurningCostAnalysis:
    """End-to-end burning cost pipeline.

    Combines :class:`LossTriangle` development, :class:`TrendEstimator`
    trending, and :class:`OnLevelPremium` on-leveling into a single
    reproducible analysis object.

    Parameters
    ----------
    triangle : LossTriangle
        Historical loss triangle.
    trend_estimator : TrendEstimator
        Fitted trend estimator for severity and/or frequency.
    on_level : OnLevelPremium
        On-level premium calculator with rate changes already applied.
    earned_premium : Mapping[int, float], optional
        Earned premium per accident year.  Required to produce loss ratios.

    Example
    -------
    >>> bc = BurningCostAnalysis(
    ...     triangle=tri, trend_estimator=trend, on_level=onlevel,
    ...     earned_premium={2022: 1_000_000, 2023: 1_100_000},
    ... )
    >>> results = bc.run(target_year=2026)
    """

    def __init__(
        self,
        triangle: LossTriangle,
        trend_estimator: TrendEstimator,
        on_level: OnLevelPremium,
        *,
        earned_premium: Mapping[int, float] | None = None,
    ) -> None:
        self.triangle = triangle
        self.trend_estimator = trend_estimator
        self.on_level = on_level
        self.earned_premium = (
            None if earned_premium is None else dict(earned_premium)
        )
        self._last_run: pd.DataFrame | None = None

    def run(
        self,
        target_year: int,
        *,
        development_method: Literal[
            "chain_ladder", "bornhuetter_ferguson"
        ] = "chain_ladder",
        a_priori: ArrayLike | None = None,
        average: _AVERAGE_METHODS = "volume",
        tail: float = 1.0,
    ) -> pd.DataFrame:
        """Execute the full burning cost pipeline.

        Returns
        -------
        pd.DataFrame
            Indexed by accident year with columns: ``ultimate_loss``,
            ``trend_factor``, ``trended_loss``, ``earned_premium``
            (if provided), ``on_level_factor``, ``on_level_premium``
            (if premium provided), and ``burning_cost`` /
            ``loss_ratio`` (if premium provided).
        """
        dev = DevelopmentFactors.from_triangle(self.triangle)
        ultimates = dev.ultimate_losses(
            method=development_method,
            a_priori=a_priori,
            average=average,
            tail=tail,
        )
        ays = list(ultimates.index)

        trend_factors = pd.Series(
            [
                self.trend_estimator.trend_factor(int(ay), int(target_year))
                for ay in ays
            ],
            index=ultimates.index,
            name="trend_factor",
        )
        trended = ultimates * trend_factors
        trended.name = "trended_loss"

        on_level_factors = self.on_level.on_level_factors(ays)
        on_level_factors.index = ultimates.index

        df = pd.DataFrame(
            {
                "ultimate_loss": ultimates.values,
                "trend_factor": trend_factors.values,
                "trended_loss": trended.values,
                "on_level_factor": on_level_factors.values,
            },
            index=ultimates.index,
        )

        if self.earned_premium is not None:
            premiums = np.array(
                [float(self.earned_premium.get(int(ay), np.nan)) for ay in ays]
            )
            on_leveled = premiums * on_level_factors.values
            df["earned_premium"] = premiums
            df["on_level_premium"] = on_leveled
            with np.errstate(divide="ignore", invalid="ignore"):
                df["loss_ratio"] = np.where(
                    on_leveled > 0, trended.values / on_leveled, np.nan
                )
            df["burning_cost"] = df["loss_ratio"]

        self._last_run = df
        return df

    def summary(self) -> pd.DataFrame:
        """Return a summary table of the most recent :meth:`run`."""
        if self._last_run is None:
            raise RuntimeError("Call run() before summary().")
        return self._last_run.copy()
