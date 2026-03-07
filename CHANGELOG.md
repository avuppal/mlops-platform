# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-07

### Added
- `ExperimentTracker` — file-backed JSON experiment tracking with per-step metric history,
  param logging, best-run selection, and multi-run comparison.
- `ModelRegistry` — versioned model registry with Staging → Production → Archived promotion
  pipeline, production guard (at most one Production version), and one-click rollback.
- `ABTest` — traffic-splitting A/B test manager with Welch's t-test (implemented from scratch
  using `math` stdlib only), two-tailed p-value, lift calculation, and winner declaration.
- `DriftMonitor` — multi-feature drift monitor combining Population Stability Index (PSI)
  and two-sample Kolmogorov-Smirnov test with configurable thresholds.
- `population_stability_index()` — standalone PSI function with severity labels.
- `ks_drift_score()` — two-sample KS test with scipy fallback to pure-Python implementation.
- `RetrainingPipeline` — orchestrates full train → eval → register → promote loop with
  drift-based and metric-threshold-based retraining triggers.
- `configs/platform_config.yaml` — central configuration for all thresholds and paths.
- `docs/architecture.md` — four Mermaid system architecture diagrams.
- 40+ unit tests covering all components, edge cases, and persistence.
