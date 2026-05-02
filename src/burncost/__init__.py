"""burncost: burning cost analysis for P&C pricing.

Loss trending, development factors, and premium on-leveling in a single
pipeline.
"""

from burncost import diagnostics
from burncost.development import DevelopmentFactors
from burncost.onlevel import OnLevelPremium
from burncost.pipeline import BurningCostAnalysis
from burncost.trending import TrendEstimator
from burncost.triangle import LossTriangle

__version__ = "0.1.2"

__all__ = [
    "BurningCostAnalysis",
    "DevelopmentFactors",
    "LossTriangle",
    "OnLevelPremium",
    "TrendEstimator",
]
