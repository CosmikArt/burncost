"""Trend estimation for severity and frequency.

Seven functional forms are supported (exponential, linear, multiplicative,
additive, power, log_linear, mixed). The fit is OLS in log-space for the
multiplicative families and direct OLS for the additive ones.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike


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

    * **exponential**: ``y = a * exp(b * t)``
    * **linear** (``y = a + b * t``)
    * **multiplicative**. ``y_{t} = y_{t-1} * (1 + r)``
    * **additive**: ``y_{t} = y_{t-1} + d``
    * **power** (``y = a * t^b``)
    * **log_linear** fits ``ln(y) = a + b * t`` by OLS in log-space.
    * **mixed**: equal-weighted blend of exponential and linear fits.

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
        # Mean of fit-time x. Used to evaluate centered-fit families
        # (exponential, multiplicative, log_linear, mixed-exp side) at
        # predict time. Default 0.0 keeps manually-injected estimators
        # (without a fit() call) backward-compatible.
        self._x_mean: float = 0.0

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

        # Center x around its mean before polyfit. With raw calendar
        # years the Vandermonde matrix can be ill-conditioned; centering
        # is mathematically equivalent and numerically much better.
        x_mean = float(x_arr.mean())
        x_centered = x_arr - x_mean
        self._x_mean = x_mean

        if method in ("exponential", "multiplicative", "log_linear"):
            if np.any(y_arr <= 0):
                raise ValueError(
                    f"{method} trend requires strictly positive y values."
                )
            ln_y = np.log(y_arr)
            slope, intercept = np.polyfit(x_centered, ln_y, 1)
            # Store centered-fit params; predict uses (x - x_mean).
            self._params = {"a": float(np.exp(intercept)), "b": float(slope)}
        elif method in ("linear", "additive"):
            slope, intercept = np.polyfit(x_centered, y_arr, 1)
            # Back-transform to original scale so params and predict
            # behave identically to the pre-centering implementation.
            #   y = a_c + b_c*(x - xm) = (a_c - b_c*xm) + b_c*x
            a_orig = float(intercept) - float(slope) * x_mean
            self._params = {"a": a_orig, "b": float(slope)}
        elif method == "power":
            if np.any(x_arr <= 0) or np.any(y_arr <= 0):
                raise ValueError(
                    "power trend requires strictly positive x and y values."
                )
            # log(x) for typical inputs stays well-conditioned, so
            # centering on log(x) buys nothing here.
            slope, intercept = np.polyfit(np.log(x_arr), np.log(y_arr), 1)
            self._params = {"a": float(np.exp(intercept)), "b": float(slope)}
        else:  # method == "mixed" (validated against _VALID_METHODS above)
            if np.any(y_arr <= 0):
                raise ValueError(
                    "mixed trend requires strictly positive y values."
                )
            ln_y = np.log(y_arr)
            slope_e, int_e = np.polyfit(x_centered, ln_y, 1)
            slope_l, int_l = np.polyfit(x_centered, y_arr, 1)
            # Exp side stored centered (predict uses x - x_mean); linear
            # side back-transformed so its (a, b) read on the original scale.
            a_lin_orig = float(int_l) - float(slope_l) * x_mean
            self._params = {
                "a_exp": float(np.exp(int_e)),
                "b_exp": float(slope_e),
                "a_lin": a_lin_orig,
                "b_lin": float(slope_l),
            }
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict trended values at the given points."""
        if self._method is None or self._params is None:
            raise RuntimeError("Call fit() before predict().")
        x_arr = np.asarray(x, dtype=float)
        xm = self._x_mean
        p = self._params
        if self._method in ("exponential", "multiplicative", "log_linear"):
            return p["a"] * np.exp(p["b"] * (x_arr - xm))
        if self._method in ("linear", "additive"):
            return p["a"] + p["b"] * x_arr
        if self._method == "power":
            return p["a"] * np.power(x_arr, p["b"])
        # mixed
        exp_pred = p["a_exp"] * np.exp(p["b_exp"] * (x_arr - xm))
        lin_pred = p["a_lin"] + p["b_lin"] * x_arr
        return 0.5 * (exp_pred + lin_pred)

    def trend_factor(self, from_year: float, to_year: float) -> float:
        """Cumulative multiplicative trend factor between two points."""
        if self._method is None:
            raise RuntimeError("Call fit() before trend_factor().")
        pred = self.predict(np.array([from_year, to_year], dtype=float))
        if abs(pred[0]) < 1e-9:
            raise ZeroDivisionError(
                f"Predicted base value is too close to zero ({pred[0]:.2e}); "
                f"trend factor is undefined."
            )
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
