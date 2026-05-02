"""Diagnostics for trend fits and chain-ladder development.

Numeric summaries (R^2, RMSE, AIC/BIC, Mack residuals) plus a couple of
matplotlib helpers (trend residual scatter, age-to-age line plot).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from burncost.development import DevelopmentFactors
from burncost.trending import TrendEstimator


# Number of free parameters per trend method (used by AIC/BIC).
_N_PARAMS: dict[str, int] = {
    "linear": 2,
    "exponential": 2,
    "multiplicative": 1,
    "additive": 1,
    "power": 2,
    "log_linear": 2,
    "mixed": 4,
}


def _require_fitted_trend(estimator: TrendEstimator) -> tuple[np.ndarray, np.ndarray, str]:
    """Pull (x, y, method) from a fitted estimator or raise."""
    if (
        estimator.method is None
        or estimator._x is None
        or estimator._y is None
    ):
        raise RuntimeError("TrendEstimator must be fit before running diagnostics.")
    return estimator._x, estimator._y, estimator.method


def trend_fit_summary(estimator: TrendEstimator) -> dict[str, float | int | str]:
    """Summary statistics of a fitted :class:`TrendEstimator`.

    Returns
    -------
    dict
        Keys: ``method``, ``n_observations``, ``r_squared``, ``rmse``, ``mae``,
        ``residual_std``.
    """
    x, y, method = _require_fitted_trend(estimator)
    y_hat = estimator.predict(x)
    residuals = y - y_hat
    n = int(y.size)
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
    return {
        "method": method,
        "n_observations": n,
        "r_squared": float(r_squared),
        "rmse": float(math.sqrt(rss / n)),
        "mae": float(np.mean(np.abs(residuals))),
        "residual_std": float(np.std(residuals, ddof=0)),
    }


def trend_residual_plot(estimator: TrendEstimator, ax: Any = None) -> Any:
    """Scatter plot of fit residuals vs. x, with a horizontal zero line."""
    import matplotlib.pyplot as plt

    x, y, method = _require_fitted_trend(estimator)
    residuals = y - estimator.predict(x)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(x, residuals, color="black")
    ax.axhline(0.0, color="grey", linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("residual")
    ax.set_title(f"Residuals: {method}")
    return ax


def development_stability(dev: DevelopmentFactors) -> pd.DataFrame:
    """Per-interval link-ratio stability statistics.

    For each age-to-age interval, return mean, median, std, min, max, and CV
    (std / mean) of the observed link ratios. Useful for spotting unstable
    development periods.

    Returns
    -------
    pd.DataFrame
        Indexed by interval label (e.g. ``"12-24"``) with columns
        ``mean``, ``median``, ``std``, ``min``, ``max``, ``cv``.
    """
    cum = dev._triangle.values
    labels = dev._interval_labels
    rows: list[dict[str, float]] = []
    for j, label in enumerate(labels):
        a = cum[:, j]
        b = cum[:, j + 1]
        mask = ~(np.isnan(a) | np.isnan(b)) & (a != 0)
        if not mask.any():
            rows.append(
                {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "cv": float("nan"),
                }
            )
            continue
        ratios = b[mask] / a[mask]
        mean = float(np.mean(ratios))
        std = float(np.std(ratios, ddof=0))
        cv = std / abs(mean) if mean != 0 else float("nan")
        rows.append(
            {
                "mean": mean,
                "median": float(np.median(ratios)),
                "std": std,
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
                "cv": cv,
            }
        )
    return pd.DataFrame(
        rows,
        index=pd.Index(labels, name="interval"),
        columns=["mean", "median", "std", "min", "max", "cv"],
    )


def development_factor_plot(dev: DevelopmentFactors, ax: Any = None) -> Any:
    """One line per accident year showing its age-to-age ratios."""
    import matplotlib.pyplot as plt

    cum = dev._triangle.values
    ays = dev._triangle.accident_years
    labels = dev._interval_labels
    n_rows, n_cols = cum.shape

    if ax is None:
        _, ax = plt.subplots()
    x_pos = np.arange(len(labels))
    for i in range(n_rows):
        ratios: list[float] = []
        for j in range(n_cols - 1):
            a = cum[i, j]
            b = cum[i, j + 1]
            if not np.isnan(a) and not np.isnan(b) and a != 0:
                ratios.append(b / a)
            else:
                ratios.append(float("nan"))
        ax.plot(x_pos, ratios, marker="o", label=str(ays[i]))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("interval")
    ax.set_ylabel("age-to-age factor")
    ax.legend(title="AY", fontsize="small")
    return ax


def chain_ladder_residuals(dev: DevelopmentFactors) -> pd.DataFrame:
    """Mack-style standardized chain-ladder residuals.

    For each observed cell ``(i, j+1)``::

        residual = (C_{i,j+1} - f_j * C_{i,j}) / sqrt(C_{i,j})

    where ``f_j`` is the volume-weighted age-to-age factor. Cells that are
    not jointly observed (or have ``C_{i,j} <= 0``) are returned as NaN.

    See Mack, T. (1993).
    """
    cum = dev._triangle.values
    ays = dev._triangle.accident_years
    labels = dev._interval_labels
    n_rows, n_cols = cum.shape

    selected = dev.selected_factors(method="volume").values
    out = np.full((n_rows, n_cols - 1), np.nan)
    for j in range(n_cols - 1):
        f_j = float(selected[j])
        if not np.isfinite(f_j):
            continue
        for i in range(n_rows):
            c_ij = cum[i, j]
            c_ij1 = cum[i, j + 1]
            if (
                np.isnan(c_ij)
                or np.isnan(c_ij1)
                or c_ij <= 0
            ):
                continue
            out[i, j] = (c_ij1 - f_j * c_ij) / math.sqrt(c_ij)
    return pd.DataFrame(
        out,
        index=pd.Index(ays, name="accident_year"),
        columns=labels,
    )


def goodness_of_fit(estimator: TrendEstimator) -> dict[str, float]:
    """AIC, BIC, log-likelihood, and R^2 under a Gaussian residual model.

    Computed as::

        loglik = -n/2 * (log(2*pi*rss/n) + 1)
        AIC    = -2*loglik + 2*k
        BIC    = -2*loglik + k*log(n)

    where ``k`` is the number of fitted parameters for the chosen method.
    """
    x, y, method = _require_fitted_trend(estimator)
    n = int(y.size)
    y_hat = estimator.predict(x)
    residuals = y - y_hat
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
    k = _N_PARAMS[method]
    if rss <= 0 or n <= 0:
        loglik = float("inf")
    else:
        loglik = -0.5 * n * (math.log(2.0 * math.pi * rss / n) + 1.0)
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * math.log(n)
    return {
        "loglik": float(loglik),
        "aic": float(aic),
        "bic": float(bic),
        "r_squared": float(r_squared),
    }
