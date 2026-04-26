"""Core classes for burning cost analysis.

This module provides the foundational building blocks for a complete
burning-cost workflow: loss triangles, development factors, trend
estimation, premium on-leveling, and an end-to-end analysis pipeline.

Typical usage
-------------
>>> tri = LossTriangle(data, accident_years, dev_periods)
>>> dev = DevelopmentFactors.from_triangle(tri)
>>> trend = TrendEstimator()
>>> trend.fit(years, severity, method="exponential")
>>> onlevel = OnLevelPremium()
>>> onlevel.apply_rate_changes(rate_changes_df)
>>> bc = BurningCostAnalysis(tri, trend, onlevel)
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Loss triangle container
# ---------------------------------------------------------------------------

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
        # cumsum propagates NaN naturally — once NaN appears, the rest is NaN.
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


# ---------------------------------------------------------------------------
# Trend estimation
# ---------------------------------------------------------------------------

_TREND_METHODS = Literal[
    "exponential",
    "linear",
    "multiplicative",
    "additive",
    "power",
    "log_linear",
    "mixed",
]

_VALID_METHODS = {
    "exponential",
    "linear",
    "multiplicative",
    "additive",
    "power",
    "log_linear",
    "mixed",
}


class TrendEstimator:
    """Estimate frequency or severity trend from historical data.

    Supports seven functional forms commonly used in actuarial trending:

    * **exponential** — ``y = a * exp(b * t)``
    * **linear** — ``y = a + b * t``
    * **multiplicative** — ``y_{t} = y_{t-1} * (1 + r)``
    * **additive** — ``y_{t} = y_{t-1} + d``
    * **power** — ``y = a * t^b``
    * **log_linear** — ``ln(y) = a + b * t``  (OLS in log-space)
    * **mixed** — equal-weighted blend of exponential and linear fits

    Workflow
    --------
    1. Instantiate: ``trend = TrendEstimator()``
    2. Fit: ``trend.fit(years, values, method="exponential")``
    3. Query: ``trend.trend_factor(from_year, to_year)``
    4. Visualise: ``trend.plot()``

    References
    ----------
    Werner & Modlin, *Basic Ratemaking*, Chapter 4.
    """

    def __init__(self) -> None:
        self._method: str | None = None
        self._params: dict[str, float] | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    # -- core API -----------------------------------------------------------

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        method: _TREND_METHODS = "exponential",
    ) -> "TrendEstimator":
        """Fit a trend curve to observed data."""
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown trend method {method!r}. Valid options: "
                f"{sorted(_VALID_METHODS)}."
            )

        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape.")
        if x_arr.size < 2:
            raise ValueError("At least two observations are required.")
        if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
            raise ValueError("x and y must contain only finite values.")

        self._method = method
        self._x = x_arr
        self._y = y_arr

        if method in ("exponential", "multiplicative", "log_linear"):
            if np.any(y_arr <= 0):
                raise ValueError(
                    f"{method} trend requires strictly positive y values."
                )
            ln_y = np.log(y_arr)
            slope, intercept = np.polyfit(x_arr, ln_y, 1)
            self._params = {"a": float(np.exp(intercept)), "b": float(slope)}
        elif method in ("linear", "additive"):
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            self._params = {"a": float(intercept), "b": float(slope)}
        elif method == "power":
            if np.any(x_arr <= 0) or np.any(y_arr <= 0):
                raise ValueError(
                    "power trend requires strictly positive x and y values."
                )
            slope, intercept = np.polyfit(np.log(x_arr), np.log(y_arr), 1)
            self._params = {"a": float(np.exp(intercept)), "b": float(slope)}
        else:  # method == "mixed" (validated against _VALID_METHODS above)
            if np.any(y_arr <= 0):
                raise ValueError(
                    "mixed trend requires strictly positive y values."
                )
            ln_y = np.log(y_arr)
            slope_e, int_e = np.polyfit(x_arr, ln_y, 1)
            slope_l, int_l = np.polyfit(x_arr, y_arr, 1)
            self._params = {
                "a_exp": float(np.exp(int_e)),
                "b_exp": float(slope_e),
                "a_lin": float(int_l),
                "b_lin": float(slope_l),
            }
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict trended values at the given points."""
        if self._method is None or self._params is None:
            raise RuntimeError("Call fit() before predict().")
        x_arr = np.asarray(x, dtype=float)
        p = self._params
        if self._method in ("exponential", "multiplicative", "log_linear"):
            return p["a"] * np.exp(p["b"] * x_arr)
        if self._method in ("linear", "additive"):
            return p["a"] + p["b"] * x_arr
        if self._method == "power":
            return p["a"] * np.power(x_arr, p["b"])
        # mixed
        exp_pred = p["a_exp"] * np.exp(p["b_exp"] * x_arr)
        lin_pred = p["a_lin"] + p["b_lin"] * x_arr
        return 0.5 * (exp_pred + lin_pred)

    def trend_factor(self, from_year: float, to_year: float) -> float:
        """Cumulative multiplicative trend factor between two points."""
        if self._method is None:
            raise RuntimeError("Call fit() before trend_factor().")
        pred = self.predict(np.array([from_year, to_year], dtype=float))
        if pred[0] == 0:
            raise ZeroDivisionError("Predicted base value is zero.")
        return float(pred[1] / pred[0])

    @property
    def params(self) -> dict[str, float]:
        """Return a copy of the fitted parameters."""
        if self._params is None:
            raise RuntimeError("Call fit() before accessing params.")
        return dict(self._params)

    @property
    def method(self) -> str | None:
        """Name of the fitted trend method, or ``None`` if not yet fit."""
        return self._method

    def annual_rate(self) -> float:
        """Return the implied annual trend rate (e.g. 0.05 for +5%/yr).

        For multiplicative methods the rate is ``exp(b) - 1``; for additive
        methods it is the slope expressed as a fraction of the mean of the
        observed data.
        """
        if self._method is None or self._params is None:
            raise RuntimeError("Call fit() before annual_rate().")
        p = self._params
        if self._method in ("exponential", "multiplicative", "log_linear"):
            return float(np.exp(p["b"]) - 1.0)
        if self._method in ("linear", "additive"):
            mean_y = float(np.mean(self._y))  # type: ignore[arg-type]
            if mean_y == 0:
                return 0.0
            return float(p["b"] / mean_y)
        if self._method == "power":
            # No constant annual rate for power; report instantaneous at mean(x).
            mean_x = float(np.mean(self._x))  # type: ignore[arg-type]
            return float(p["b"] / mean_x)
        # mixed
        return 0.5 * (
            float(np.exp(p["b_exp"]) - 1.0)
            + (
                p["b_lin"] / float(np.mean(self._y))  # type: ignore[arg-type]
                if np.mean(self._y) != 0  # type: ignore[arg-type]
                else 0.0
            )
        )

    def plot(self, *, ax: Any = None) -> Any:
        """Plot observed data, fitted curve, and a short extrapolation."""
        if self._method is None:
            raise RuntimeError("Call fit() before plot().")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        x_obs = self._x
        y_obs = self._y
        assert x_obs is not None and y_obs is not None
        x_min, x_max = float(x_obs.min()), float(x_obs.max())
        span = x_max - x_min if x_max > x_min else 1.0
        x_grid = np.linspace(x_min, x_max + 0.5 * span, 100)
        ax.scatter(x_obs, y_obs, color="black", label="observed")
        ax.plot(x_grid, self.predict(x_grid), label=f"{self._method} fit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        return ax


# ---------------------------------------------------------------------------
# Development factors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Premium on-leveling
# ---------------------------------------------------------------------------

def _decimal_year(date: Any) -> float:
    """Convert a date to a decimal year, accounting for leap years."""
    ts = pd.Timestamp(date)
    year_start = pd.Timestamp(year=ts.year, month=1, day=1)
    year_end = pd.Timestamp(year=ts.year + 1, month=1, day=1)
    frac = (ts - year_start) / (year_end - year_start)
    return float(ts.year + frac)


def _earned_density(s: float, year: float, term: float) -> float:
    """Fraction of a unit-rate policy issued at *s* earned during AY *year*."""
    return max(0.0, min(s + term, year + 1) - max(s, year))


def _earned_integral(a: float, b: float, year: float, term: float) -> float:
    """Exact integral of the piecewise-linear earned density from *a* to *b*.

    The density is piecewise linear with breakpoints at
    ``{year - term, year + 1 - term, year, year + 1}``; trapezoidal
    integration on those points is therefore exact.
    """
    a = max(a, year - term)
    b = min(b, year + 1)
    if a >= b:
        return 0.0
    pts = sorted({a, b, year - term, year + 1 - term, year, year + 1})
    pts = [p for p in pts if a <= p <= b]
    total = 0.0
    for s1, s2 in zip(pts[:-1], pts[1:]):
        h1 = _earned_density(s1, year, term)
        h2 = _earned_density(s2, year, term)
        total += 0.5 * (h1 + h2) * (s2 - s1)
    return total


class OnLevelPremium:
    """On-level historical premiums using the parallelogram method.

    Adjusts earned premiums so every historical period reflects the *current*
    rate level, enabling apples-to-apples loss-ratio comparisons.

    Workflow
    --------
    1. Supply a rate-change history via :meth:`apply_rate_changes`.
    2. Compute on-level factors with :meth:`on_level_factor` or apply the
       full parallelogram method with :meth:`parallelogram`.

    References
    ----------
    Werner & Modlin, *Basic Ratemaking*, Chapter 5.
    """

    def __init__(self) -> None:
        self._rate_changes: pd.DataFrame | None = None
        self._current_level: float = 1.0

    def apply_rate_changes(self, rate_changes: pd.DataFrame) -> "OnLevelPremium":
        """Ingest a schedule of historical rate changes."""
        if not isinstance(rate_changes, pd.DataFrame):
            raise TypeError("rate_changes must be a pandas DataFrame.")
        required = {"effective_date", "change"}
        missing = required - set(rate_changes.columns)
        if missing:
            raise ValueError(
                f"rate_changes is missing required columns: {sorted(missing)}"
            )
        df = rate_changes[["effective_date", "change"]].copy()
        df["effective_date"] = pd.to_datetime(df["effective_date"])
        df["change"] = df["change"].astype(float)
        if df["change"].isna().any():
            raise ValueError("rate_changes 'change' column must not contain NaN.")
        if (df["change"] <= -1.0).any():
            raise ValueError(
                "rate_changes 'change' values must be greater than -1 "
                "(a -100% change would zero the rate level)."
            )
        df = df.sort_values("effective_date").reset_index(drop=True)
        df["cum_factor"] = (1.0 + df["change"]).cumprod()
        self._rate_changes = df
        self._current_level = (
            float(df["cum_factor"].iloc[-1]) if len(df) else 1.0
        )
        return self

    @property
    def current_level(self) -> float:
        """Cumulative rate level after applying all rate changes."""
        return self._current_level

    @property
    def rate_changes(self) -> pd.DataFrame | None:
        """Sorted rate-change schedule (or ``None`` if not provided)."""
        return None if self._rate_changes is None else self._rate_changes.copy()

    def parallelogram(
        self,
        policy_year: int,
        *,
        term_months: int = 12,
    ) -> float:
        """Compute the on-level factor using the parallelogram method."""
        if self._rate_changes is None or len(self._rate_changes) == 0:
            return 1.0
        if term_months <= 0:
            raise ValueError("term_months must be positive.")

        Y = float(policy_year)
        T = term_months / 12.0

        dates = [_decimal_year(d) for d in self._rate_changes["effective_date"]]
        cum_factors = self._rate_changes["cum_factor"].astype(float).tolist()

        # Build segments: (lo, hi, level)
        segments: list[tuple[float, float, float]] = []
        segments.append((-np.inf, dates[0], 1.0))
        for i in range(len(dates) - 1):
            segments.append((dates[i], dates[i + 1], cum_factors[i]))
        segments.append((dates[-1], np.inf, cum_factors[-1]))

        weighted = 0.0
        total = 0.0
        for lo, hi, level in segments:
            w = _earned_integral(lo, hi, Y, T)
            weighted += w * level
            total += w

        # `total` is the integral of the earned density across the full real
        # line clamped to [Y - T, Y + 1] — always equal to T > 0.  And every
        # rate level is strictly positive (changes <= -100% are rejected),
        # so the weighted average is positive.
        avg_level = weighted / total
        return self._current_level / avg_level

    def on_level_factor(self, policy_year: int) -> float:
        """Return the on-level factor for a single policy year (annual term)."""
        return self.parallelogram(policy_year, term_months=12)

    def on_level_factors(
        self,
        policy_years: ArrayLike,
        *,
        term_months: int = 12,
    ) -> pd.Series:
        """Vectorised :meth:`parallelogram` over a list of policy years."""
        years = list(np.asarray(policy_years, dtype=int).ravel())
        return pd.Series(
            [self.parallelogram(y, term_months=term_months) for y in years],
            index=pd.Index(years, name="policy_year"),
            name="on_level_factor",
        )


# ---------------------------------------------------------------------------
# End-to-end burning cost pipeline
# ---------------------------------------------------------------------------

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
