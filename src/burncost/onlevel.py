"""Premium on-leveling via the parallelogram method (Werner & Modlin, ch. 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


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
        # line clamped to [Y - T, Y + 1], always equal to T > 0.  And every
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
