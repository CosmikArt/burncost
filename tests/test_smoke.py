"""Smoke tests: verify that the package imports and classes instantiate."""

import numpy as np
import pytest


class TestImports:
    """Ensure top-level imports resolve without errors."""

    def test_import_package(self):
        import burncost
        assert hasattr(burncost, "__version__")

    def test_import_classes(self):
        from burncost import (
            BurningCostAnalysis,
            DevelopmentFactors,
            LossTriangle,
            OnLevelPremium,
            TrendEstimator,
        )


class TestInstantiation:
    """Ensure core classes can be instantiated with minimal arguments."""

    def test_loss_triangle(self):
        from burncost import LossTriangle

        data = np.array([[100, 150], [110, np.nan]])
        tri = LossTriangle(data, accident_years=[2023, 2024], dev_periods=[12, 24])
        assert tri.shape == (2, 2)
        assert tri.is_cumulative is True

    def test_loss_triangle_repr(self):
        from burncost import LossTriangle

        data = np.array([[100, 150], [110, np.nan]])
        tri = LossTriangle(data, accident_years=[2023, 2024], dev_periods=[12, 24])
        assert "LossTriangle" in repr(tri)

    def test_loss_triangle_validation(self):
        from burncost import LossTriangle

        with pytest.raises(ValueError):
            LossTriangle(np.array([1, 2, 3]), [2023], [12, 24, 36])

    def test_trend_estimator(self):
        from burncost import TrendEstimator

        trend = TrendEstimator()
        assert trend._method is None

    def test_development_factors(self):
        from burncost import DevelopmentFactors, LossTriangle

        data = np.array([[100, 150], [110, np.nan]])
        tri = LossTriangle(data, [2023, 2024], [12, 24])
        dev = DevelopmentFactors.from_triangle(tri)
        assert dev._triangle is tri

    def test_on_level_premium(self):
        from burncost import OnLevelPremium

        olp = OnLevelPremium()
        assert olp._rate_changes is None

    def test_burning_cost_analysis(self):
        from burncost import (
            BurningCostAnalysis,
            LossTriangle,
            OnLevelPremium,
            TrendEstimator,
        )

        data = np.array([[100, 150], [110, np.nan]])
        tri = LossTriangle(data, [2023, 2024], [12, 24])
        trend = TrendEstimator()
        olp = OnLevelPremium()
        bc = BurningCostAnalysis(triangle=tri, trend_estimator=trend, on_level=olp)
        assert bc.triangle is tri
