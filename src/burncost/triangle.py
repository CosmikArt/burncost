"""Loss triangle container."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class LossTriangle:
    """Container for incremental or cumulative loss triangles.

    A loss triangle organises historical claims by **accident year** (rows) and
    **development period** (columns).  This class stores the numeric data and
    provides convenience properties for downstream consumers such as
    :class:`DevelopmentFactors`.

    Parameters
    ----------
    data : array-like, shape (n_years, n_periods)
        Two-dimensional array of loss amounts.  Missing / not-yet-observed
        cells should be ``np.nan``.
    accident_years : list[int]
        Labels for each row (e.g. ``[2019, 2020, 2021, 2022]``).
    dev_periods : list[int]
        Labels for each column in months or years
        (e.g. ``[12, 24, 36, 48]``).
    cumulative : bool, default True
        Whether *data* represents cumulative losses.  If ``False`` the
        triangle is treated as incremental.

    Attributes
    ----------
    values : np.ndarray
    accident_years : list[int]
    dev_periods : list[int]
    is_cumulative : bool
    """

    def __init__(
        self,
        data: ArrayLike,
        accident_years: list[int],
        dev_periods: list[int],
        cumulative: bool = True,
    ) -> None:
        self.values: np.ndarray = np.asarray(data, dtype=float)
        self.accident_years = list(accident_years)
        self.dev_periods = list(dev_periods)
        self.is_cumulative = cumulative

        if self.values.ndim != 2:
            raise ValueError("data must be two-dimensional.")
        if self.values.shape[0] != len(self.accident_years):
            raise ValueError(
                "Number of rows in data must equal len(accident_years)."
            )
        if self.values.shape[1] != len(self.dev_periods):
            raise ValueError(
                "Number of columns in data must equal len(dev_periods)."
            )

    # -- conversions --------------------------------------------------------

    def to_cumulative(self) -> "LossTriangle":
        """Return a cumulative version of this triangle.

        If the triangle is already cumulative the same instance is returned.

        Returns
        -------
        LossTriangle
        """
        if self.is_cumulative:
            return self
        # cumsum propagates NaN naturally: once NaN appears, the rest is NaN.
        cum = np.cumsum(self.values, axis=1)
        return LossTriangle(
            cum,
            accident_years=self.accident_years,
            dev_periods=self.dev_periods,
            cumulative=True,
        )

    def to_incremental(self) -> "LossTriangle":
        """Return an incremental version of this triangle.

        Returns
        -------
        LossTriangle
        """
        if not self.is_cumulative:
            return self
        first = self.values[:, :1]
        rest = np.diff(self.values, axis=1)
        inc = np.concatenate([first, rest], axis=1)
        return LossTriangle(
            inc,
            accident_years=self.accident_years,
            dev_periods=self.dev_periods,
            cumulative=False,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the triangle as a :class:`pandas.DataFrame`.

        Rows are indexed by accident year and columns by development period.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(
            self.values,
            index=pd.Index(self.accident_years, name="accident_year"),
            columns=pd.Index(self.dev_periods, name="dev_period"),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """(n_accident_years, n_dev_periods)."""
        return self.values.shape  # type: ignore[return-value]

    @property
    def latest_diagonal(self) -> pd.Series:
        """Most recent observed value per accident year.

        Returns
        -------
        pd.Series
            Indexed by accident year.
        """
        latest = np.full(self.values.shape[0], np.nan)
        for i in range(self.values.shape[0]):
            row = self.values[i]
            valid = np.where(~np.isnan(row))[0]
            if len(valid):
                latest[i] = row[valid[-1]]
        return pd.Series(latest, index=self.accident_years, name="latest")

    def __repr__(self) -> str:
        return (
            f"LossTriangle(years={self.accident_years}, "
            f"periods={self.dev_periods}, "
            f"cumulative={self.is_cumulative})"
        )
