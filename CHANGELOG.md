# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-05-01

### Added

- `burncost.diagnostics` module: `trend_fit_summary`, `goodness_of_fit`
  (AIC/BIC/log-likelihood under a Gaussian residual model),
  `trend_residual_plot`, `development_stability` (per-interval link-ratio
  mean/median/std/min/max/CV), `development_factor_plot`, and Mack-style
  `chain_ladder_residuals`.
- 15 tests covering the diagnostics surface; total suite at 115 tests.

### Changed

- Package split from a single `core.py` into per-concern modules:
  `triangle`, `development`, `trending`, `onlevel`, `pipeline`,
  `diagnostics`. The five public classes still import from the package
  root.
- Tightened README and docstrings. Module list now reflects the actual
  package layout.
- Bumped development status to Beta.

## [0.0.1] - 2026-04-26

### Added

- `LossTriangle` container for incremental/cumulative loss triangles, with
  `to_cumulative`, `to_incremental`, `to_dataframe`, and `latest_diagonal`
  helpers.
- `TrendEstimator` with seven trending methods (exponential, linear,
  multiplicative, additive, power, log-linear, mixed), plus `predict`,
  `trend_factor`, `annual_rate`, and `plot`.
- `DevelopmentFactors` for age-to-age and age-to-ultimate factor computation
  (volume / simple / medial averaging; exponential tail extrapolation;
  chain-ladder and Bornhuetter-Ferguson ultimate-loss projection).
- `OnLevelPremium` for premium on-leveling via the parallelogram method,
  with exact piecewise-linear earned-premium integration, a vectorised
  `on_level_factors` helper, and rate-change history ingestion.
- `BurningCostAnalysis` end-to-end pipeline wiring development, trending,
  and on-leveling into a single output table.
- Test suite (100 tests) at 100% line and branch coverage.
