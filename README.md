[![PyPI version](https://img.shields.io/pypi/v/burncost?color=blue)](https://pypi.org/project/burncost/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)]()

# burncost

Python library for ratemaking-ready loss-experience preparation. Loss development (chain-ladder, Bornhuetter-Ferguson), trend estimation, and premium on-leveling via the parallelogram method. Designed to chain into a downstream rate model.

## Installation

```bash
pip install burncost
```

From source:

```bash
git clone https://github.com/CosmikArt/burncost.git
cd burncost
pip install -e ".[dev]"
```

## Quickstart

```python
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove if you want an interactive window

import numpy as np
import pandas as pd
from burncost import (
    LossTriangle,
    DevelopmentFactors,
    TrendEstimator,
    OnLevelPremium,
    BurningCostAnalysis,
)

# --- 1. Loss development ---------------------------------------------------
# Cumulative paid-loss triangle (AY x development month)
triangle_data = np.array([
    [1_000, 1_500, 1_800, 1_900],
    [1_100, 1_650, 1_980,   np.nan],
    [1_200, 1_800,   np.nan, np.nan],
    [1_300,   np.nan, np.nan, np.nan],
])
accident_years = [2022, 2023, 2024, 2025]
dev_periods    = [12, 24, 36, 48]

tri = LossTriangle(triangle_data, accident_years, dev_periods)
dev = DevelopmentFactors.from_triangle(tri)

ata   = dev.age_to_age()           # age-to-age factors per interval
atu   = dev.age_to_ultimate()      # cumulative to ultimate
sel   = dev.selected_factors()     # weighted-average selected factors
tail  = dev.tail_factor()          # tail beyond last observed period

# --- 2. Trend estimation ---------------------------------------------------
years     = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
severity  = np.array([4200, 4400, 4550, 4800, 5100, 5350, 5600, 5900])

trend = TrendEstimator()
trend.fit(years, severity, method="exponential")
annual_trend  = trend.trend_factor(from_year=2022, to_year=2026)
projected     = trend.predict(np.array([2026, 2027]))
trend.plot()

# --- 3. Premium on-leveling -----------------------------------------------
rate_changes = pd.DataFrame({
    "effective_date": pd.to_datetime(["2021-07-01", "2023-01-01", "2024-04-01"]),
    "change":        [0.05, 0.08, -0.03],
})

onlevel = OnLevelPremium()
onlevel.apply_rate_changes(rate_changes)
olf = onlevel.on_level_factor(policy_year=2022)

# --- 4. End-to-end burning cost pipeline -----------------------------------
bc = BurningCostAnalysis(
    triangle=tri,
    trend_estimator=trend,
    on_level=onlevel,
)
# bc.run() will wire development -> trending -> on-leveling into one table.

# --- 5. Diagnostics --------------------------------------------------------
from burncost.diagnostics import trend_fit_summary, development_stability
print(trend_fit_summary(trend))
print(development_stability(dev))
```

## Modules

The five top-level classes are re-exported from the package root, so
`from burncost import X` works for all of them. Internally each concern
lives in its own module.

| Module | What's in it |
|---|---|
| `burncost.triangle` | `LossTriangle`: incremental/cumulative container with `to_cumulative`, `to_incremental`, `to_dataframe`, `latest_diagonal`. |
| `burncost.development` | `DevelopmentFactors`: age-to-age factors (volume / simple / medial averaging), age-to-ultimate, chain-ladder, Bornhuetter-Ferguson, exponential tail extrapolation. |
| `burncost.trending` | `TrendEstimator`: seven trend forms (exponential, linear, multiplicative, additive, power, log-linear, mixed); `predict`, `trend_factor`, `annual_rate`, `plot`. |
| `burncost.onlevel` | `OnLevelPremium`: parallelogram-method on-leveling with rate-change ingestion, vectorised `on_level_factors`, per-year `on_level_factor`. |
| `burncost.pipeline` | `BurningCostAnalysis`: end-to-end wiring of development, trending, and on-leveling into one output table. |
| `burncost.diagnostics` | `trend_fit_summary`, `goodness_of_fit`, `trend_residual_plot`, `development_stability`, `development_factor_plot`, `chain_ladder_residuals` (Mack-style). |

## References

- **Werner, G. & Modlin, C.** *Basic Ratemaking*, CAS, Chapters 4-5 (trending and loss development).
- **Mack, T.** (1993). "Distribution-free Calculation of the Standard Error of Chain Ladder Reserve Estimates." *ASTIN Bulletin*, 23(2), 213-225.
- **Bornhuetter, R. L. & Ferguson, R. E.** (1972). "The Actuary and IBNR." *Proceedings of the CAS*, LIX, 181-195.
- **Friedland, J.** *Estimating Unpaid Claims Using Basic Techniques*. Casualty Actuarial Society.

## Contributing

Run `pytest` before sending a PR.

## Author

Isaac López

MIT License. See [LICENSE](LICENSE).
