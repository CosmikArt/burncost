[![PyPI version](https://img.shields.io/pypi/v/burncost?color=blue)](https://pypi.org/project/burncost/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

# burncost

**Burning cost analysis — loss trending, development, and on-leveling for P&C pricing.**

---

## What is burncost?

Every pricing actuary knows the drill: before you can build a rate model you
need *ratemaking-ready experience*. Raw loss data must be developed to ultimate,
trended to future cost levels, and premiums must be on-leveled to a common rate
basis. This data-preparation step is the backbone of every loss-ratio or
pure-premium analysis, yet no focused Python library exists for it.

**burncost** fills that gap. It provides a single, opinionated API for the three
pillars of burning-cost analysis:

1. **Loss trending** — fit and extrapolate frequency/severity trends using seven
   well-known functional forms.
2. **Loss development** — compute age-to-age and age-to-ultimate factors from
   loss triangles via chain-ladder, Bornhuetter-Ferguson, and user-selected
   factor methods.
3. **Premium on-leveling** — bring historical premiums to current rate level
   using the parallelogram method and a schedule of rate changes.

The result is a clean, auditable pipeline that turns messy historical data into
the inputs your ratemaking model expects.

---

## Installation

### From PyPI (when published)

```bash
pip install burncost
```

### From source

```bash
git clone https://github.com/CosmikArt/burncost.git
cd burncost
pip install -e .
```

---

## Quickstart

```python
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
```

---

## Features

| Module | Capabilities |
|---|---|
| `trending` | Exponential, linear, multiplicative, additive, power, log-linear, and mixed trend fits; annual/multi-year trend factors; extrapolation |
| `development` | Age-to-age factors, weighted/simple/medial averages, chain-ladder ultimate, Bornhuetter-Ferguson, selected factors, tail-factor extrapolation |
| `onlevel` | Premium on-leveling via parallelogram method, rate-change history ingestion, on-level factors by policy or accident year |
| `diagnostics` | Trend-fit plots, development-factor stability charts, residual analysis, goodness-of-fit statistics |

---

## References

- **Werner, G. & Modlin, C.** *Basic Ratemaking*, CAS, Chapters 4-5 (trending and loss development).
- **Mack, T.** (1993). "Distribution-free Calculation of the Standard Error of Chain Ladder Reserve Estimates." *ASTIN Bulletin*, 23(2), 213-225.
- **Bornhuetter, R. L. & Ferguson, R. E.** (1972). "The Actuary and IBNR." *Proceedings of the CAS*, LIX, 181-195.
- **Friedland, J.** *Estimating Unpaid Claims Using Basic Techniques*. Casualty Actuarial Society.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes
before submitting a pull request. All code must include type hints, docstrings,
and unit tests.

---

## Author

**Isaac López**
