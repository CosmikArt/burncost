# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.0.1] - 2026-04-26

### Added

- `LossTriangle` container for incremental/cumulative loss triangles, with `to_cumulative`, `to_incremental`, `to_dataframe`, and `latest_diagonal` helpers.
- `TrendEstimator` with seven trending methods — exponential, linear, multiplicative, additive, power, log-linear, mixed — plus `predict`, `trend_factor`, `annual_rate`, and `plot`.
- `DevelopmentFactors` for age-to-age and age-to-ultimate factor computation (volume / simple / medial averaging, exponential tail extrapolation, chain-ladder and Bornhuetter-Ferguson ultimate-loss projection).
- `OnLevelPremium` for premium on-leveling via the parallelogram method, with exact piecewise-linear earned-premium integration, a vectorised `on_level_factors` helper, and rate-change history ingestion.
- `BurningCostAnalysis` end-to-end pipeline wiring development → trending → on-leveling → loss ratios.
- Test suite (100 tests) achieving 100% line and branch coverage.
