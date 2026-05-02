"""Age-to-age and age-to-ultimate development factors.

Implements the standard chain-ladder machinery (volume / simple / medial
averaging, exponential tail extrapolation, BF projection) that turns a
:class:`burncost.triangle.LossTriangle` into ultimate-loss estimates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from burncost.triangle import LossTriangle


_AVERAGE_METHODS = Literal["volume", "simple", "medial"]


def _link_ratio(numerator: np.ndarray, denominator: np.ndarray, method: str) -> float:
    """Compute a single link ratio from two paired columns."""
    valid = ~(np.isnan(numerator) | np.isnan(denominator)) & (denominator != 0)
    if not valid.any():
        return float("nan")
    a = denominator[valid]
    b = numerator[valid]
    if method == "volume":
        return float(b.sum() / a.sum())
    if method == "simple":
        return float(np.mean(b / a))
    if method == "medial":
        ratios = np.sort(b / a)
        if ratios.size > 2:
            ratios = ratios[1:-1]
        return float(np.mean(ratios))
    raise ValueError(f"Unknown averaging method: {method!r}")


class DevelopmentFactors:
    """Age-to-age and age-to-ultimate development factors.

    Compute link ratios from a :class:`LossTriangle`, select factors using
    weighted or simple averages, and project losses to ultimate via
    chain-ladder or Bornhuetter-Ferguson.

    References
    ----------
    * Mack, T. (1993). Distribution-free calculation of the standard error
      of chain ladder reserve estimates.
    * Bornhuetter, R. L. & Ferguson, R. E. (1972). The Actuary and IBNR.
    * Friedland, J. *Estimating Unpaid Claims Using Basic Techniques*. CAS.
    """

    def __init__(self, triangle: LossTriangle) -> None:
        self._triangle = triangle.to_cumulative()

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_triangle(cls, triangle: LossTriangle) -> "DevelopmentFactors":
        """Create a :class:`DevelopmentFactors` instance from a loss triangle."""
        return cls(triangle)

    # -- helpers ------------------------------------------------------------

    @property
    def _interval_labels(self) -> list[str]:
        periods = self._triangle.dev_periods
        return [f"{periods[j]}-{periods[j + 1]}" for j in range(len(periods) - 1)]

    # -- factor computation -------------------------------------------------

    def age_to_age(
        self,
        average: _AVERAGE_METHODS = "volume",
    ) -> pd.DataFrame:
        """Individual age-to-age link ratios per accident year.

        The final row, labelled ``"selected"``, contains the averaged
        factors using the chosen method.
        """
        cum = self._triangle.values
        ays = self._triangle.accident_years
        n_rows, n_cols = cum.shape
        if n_cols < 2:
            return pd.DataFrame(
                index=pd.Index(ays, name="accident_year"),
            )

        ratios = np.full((n_rows, n_cols - 1), np.nan)
        for i in range(n_rows):
            for j in range(n_cols - 1):
                a = cum[i, j]
                b = cum[i, j + 1]
                if not np.isnan(a) and not np.isnan(b) and a != 0:
                    ratios[i, j] = b / a

        labels = self._interval_labels
        df = pd.DataFrame(
            ratios,
            index=pd.Index(ays, name="accident_year"),
            columns=labels,
        )
        selected = self.selected_factors(method=average)
        df.loc["selected"] = selected.values
        return df

    def selected_factors(
        self,
        method: _AVERAGE_METHODS = "volume",
    ) -> pd.Series:
        """Return selected age-to-age factors."""
        cum = self._triangle.values
        n_rows, n_cols = cum.shape
        selected = []
        for j in range(n_cols - 1):
            selected.append(
                _link_ratio(cum[:, j + 1], cum[:, j], method)
            )
        return pd.Series(
            selected,
            index=pd.Index(self._interval_labels, name="interval"),
            name="selected",
        )

    def age_to_ultimate(
        self,
        method: _AVERAGE_METHODS = "volume",
        *,
        tail: float = 1.0,
    ) -> pd.Series:
        """Cumulative age-to-ultimate factors indexed by development period."""
        selected = self.selected_factors(method=method).values
        periods = self._triangle.dev_periods
        n = len(periods)
        atu = np.full(n, np.nan)
        atu[-1] = float(tail)
        for j in range(n - 2, -1, -1):
            atu[j] = float(selected[j]) * atu[j + 1]
        return pd.Series(
            atu,
            index=pd.Index(periods, name="dev_period"),
            name="age_to_ultimate",
        )

    def tail_factor(self, method: str = "unity") -> float:
        """Estimate a tail factor beyond the last observed development period.

        Parameters
        ----------
        method : str, default ``"unity"``
            ``"unity"`` (or ``"none"``) returns ``1.0``.  ``"exponential"``
            fits an exponential decay to ``ln(f - 1)`` for the selected
            factors that exceed unity, then accumulates the implied factors
            until they are within ``1e-6`` of one (capped at 50 steps).
        """
        if method in ("unity", "none"):
            return 1.0
        if method == "exponential":
            sel = self.selected_factors()
            ages = np.asarray(sel.index.map(lambda lbl: float(lbl.split("-")[1])))
            values = sel.values.astype(float)
            mask = np.isfinite(values) & (values > 1.0)
            if mask.sum() < 2:
                return 1.0
            excess = np.log(values[mask] - 1.0)
            ages_used = ages[mask]
            slope, intercept = np.polyfit(ages_used, excess, 1)
            if slope >= 0:
                # Trend not decaying; refuse to extrapolate.
                return 1.0
            intervals = np.diff(np.asarray(self._triangle.dev_periods, dtype=float))
            step = float(intervals.mean()) if intervals.size else 12.0
            tail = 1.0
            last_age = float(ages[-1])
            for k in range(1, 51):
                age_k = last_age + k * step
                f_k = 1.0 + float(np.exp(intercept + slope * age_k))
                if abs(f_k - 1.0) < 1e-6:
                    break
                tail *= f_k
            return tail
        raise ValueError(f"Unknown tail method: {method!r}")

    def ultimate_losses(
        self,
        method: Literal["chain_ladder", "bornhuetter_ferguson"] = "chain_ladder",
        *,
        a_priori: ArrayLike | None = None,
        average: _AVERAGE_METHODS = "volume",
        tail: float = 1.0,
    ) -> pd.Series:
        """Project losses to ultimate.

        Parameters
        ----------
        method : str
            ``"chain_ladder"`` or ``"bornhuetter_ferguson"``.
        a_priori : array-like, optional
            A-priori expected ultimate losses (length = n_accident_years),
            required for the BF method.
        average : str, default ``"volume"``
            Averaging method passed through to :meth:`selected_factors`.
        tail : float, default 1.0
            Tail factor applied beyond the last observed period.
        """
        cum = self._triangle.values
        ays = self._triangle.accident_years
        atu = self.age_to_ultimate(method=average, tail=tail).values

        n_rows = cum.shape[0]
        ultimates = np.full(n_rows, np.nan)

        a_priori_arr: np.ndarray | None = None
        if method == "bornhuetter_ferguson":
            if a_priori is None:
                raise ValueError(
                    "Bornhuetter-Ferguson method requires the a_priori argument."
                )
            a_priori_arr = np.asarray(a_priori, dtype=float).ravel()
            if a_priori_arr.size != n_rows:
                raise ValueError(
                    "a_priori must have one value per accident year "
                    f"(got {a_priori_arr.size}, expected {n_rows})."
                )
        elif method != "chain_ladder":
            raise ValueError(f"Unknown ultimate-loss method: {method!r}")

        for i in range(n_rows):
            row = cum[i]
            valid = np.where(~np.isnan(row))[0]
            if not len(valid):
                continue
            j = int(valid[-1])
            latest = float(row[j])
            atu_j = float(atu[j])
            if method == "chain_ladder":
                ultimates[i] = latest * atu_j
            else:  # bornhuetter_ferguson
                expected = float(a_priori_arr[i])  # type: ignore[index]
                pct_unreported = 1.0 - 1.0 / atu_j if atu_j != 0 else 0.0
                ultimates[i] = latest + expected * pct_unreported

        return pd.Series(
            ultimates,
            index=pd.Index(ays, name="accident_year"),
            name="ultimate",
        )
