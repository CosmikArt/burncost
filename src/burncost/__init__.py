"""burncost — Burning cost analysis for P&C pricing.

Loss trending, development factors, and premium on-leveling in one
opinionated pipeline.
"""

from burncost.core import (
    BurningCostAnalysis,
    DevelopmentFactors,
    LossTriangle,
    OnLevelPremium,
    TrendEstimator,
)

__version__ = "0.0.1"

__all__ = [
    "BurningCostAnalysis",
    "DevelopmentFactors",
    "LossTriangle",
    "OnLevelPremium",
    "TrendEstimator",
]
