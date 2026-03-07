# mlops-platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-40%2B%20passing-brightgreen.svg)](#running-the-tests)

**Production-grade MLOps platform** demonstrating enterprise patterns for experiment tracking, model registry, A/B testing, drift detection, and automated retraining pipelines — all in pure Python with **zero external MLOps dependencies**.

---

## Overview

Modern ML systems fail not at training time but at the operational layer: models degrade silently, experiments aren't reproducible, and promotions happen without statistical validation. This platform addresses each gap with a focused, testable component.

```
mlops-platform/
├── src/
│   ├── experiment.py   # Experiment tracking — run logging, metric history, param comparison
│   ├── registry.py     # Model registry — versioning, stage promotion, rollback
│   ├── ab_test.py      # A/B testing — Welch t-test, traffic splitting, statistical significance
│   ├── drift.py        # Drift detection — PSI, KS test, multi-feature DriftMonitor
│   └── pipeline.py     # Retraining pipeline — trigger conditions, train→register→promote loop
├── configs/
│   └── platform_config.yaml
├── tests/              # 40+ unit tests, 100% CPU, zero network calls
└── docs/
    └── architecture.md # 4 Mermaid system diagrams
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/avuppal/mlops-platform.git
cd mlops-platform

# Install (stdlib + numpy only)
pip install -r requirements.txt

# Run all tests
pytest -v
```

---

## Components

### 1. Experiment Tracker (`src/experiment.py`)

File-backed experiment tracking with per-step metric history and cross-run comparison.

```python
from src import ExperimentTracker

tracker = ExperimentTracker("./mlops_data/experiments")

run_id = tracker.start_run("fraud_clf_v2", params={"lr": 0.001, "epochs": 50})
tracker.log_metric(run_id, "accuracy", 0.923, step=50)
tracker.log_metric(run_id, "f1",       0.910, step=50)
tracker.end_run(run_id)

best = tracker.get_best_run("fraud_clf_v2", metric="f1", mode="max")
print(f"Best run: {best['run_id']}  F1={best['metrics']['f1'][-1]['value']:.3f}")

comparison = tracker.compare_runs([run_id])
# → {run_id: {"accuracy": 0.923, "f1": 0.910, "params": {...}}}
```

### 2. Model Registry (`src/registry.py`)

Versioned model registry with Staging → Production → Archived promotion and one-click rollback.

```python
from src import ModelRegistry

registry = ModelRegistry("./mlops_data/registry")

registry.register_model(
    name="fraud_clf",
    version="2.1.0",
    artifact_path="models/fraud_clf/2.1.0/model.pkl",
    metrics={"accuracy": 0.934, "f1": 0.921},
    run_id=run_id,
)

registry.promote("fraud_clf", "2.1.0", "Staging")
registry.promote("fraud_clf", "2.1.0", "Production")  # auto-archives previous production

prod = registry.get_production_model("fraud_clf")
print(f"Serving: v{prod['version']}  accuracy={prod['metrics']['accuracy']:.3f}")

registry.rollback("fraud_clf")  # instant recovery if new version misbehaves
```

### 3. A/B Testing (`src/ab_test.py`)

Traffic splitting with Welch's t-test (no scipy required) and statistically-grounded winner declaration.

```python
from src import ABTest

test = ABTest("checkout_optimisation", "model_v1", "model_v2", traffic_split=0.5)

# Simulate 200 user sessions
import random
for _ in range(200):
    variant = "treatment" if random.random() < 0.5 else "control"
    score = random.gauss(0.92 if variant == "treatment" else 0.88, 0.05)
    test.record_outcome(variant, score)

stats = test.get_stats()
print(f"Lift: {stats['lift_pct']:+.1f}%  p={stats['p_value']:.4f}  significant={stats['significant']}")

winner = test.declare_winner(alpha=0.05)
print(f"Winner: {winner}")  # "control" | "treatment" | "inconclusive"
```

### 4. Drift Detection (`src/drift.py`)

PSI and KS-based multi-feature drift monitoring with configurable thresholds.

```python
from src import DriftMonitor

monitor = DriftMonitor(
    reference_data={"age": training_ages, "income": training_incomes},
    psi_threshold=0.25,
    ks_threshold=0.05,
)

report = monitor.check({"age": live_ages, "income": live_incomes})
# → {
#     "age":    {"psi": 0.03, "ks_stat": 0.04, "drifted": False, "severity": "no_drift"},
#     "income": {"psi": 0.31, "ks_stat": 0.18, "drifted": True,  "severity": "significant_drift"},
#     "overall_drifted": True,
# }
```

### 5. Retraining Pipeline (`src/pipeline.py`)

Orchestrates the full loop: drift/metric check → train → eval → register → conditional promote.

```python
from src import RetrainingPipeline

pipeline = RetrainingPipeline(
    tracker=tracker,
    registry=registry,
    drift_monitor=monitor,
    train_fn=lambda data: fit_model(data),
    eval_fn=lambda model, data: evaluate(model, data),
    metric="accuracy",
    retrain_metric_threshold=0.80,
)

# Decide whether to retrain
drift_report = monitor.check(current_data)
if pipeline.should_retrain(prod_metrics, drift_report):
    result = pipeline.run("fraud_clf", train_data, eval_data)
    print(f"Retrained v{result['version']}  promoted={result['promoted']}")
```

---

## Capability Comparison

| Capability | This platform | MLflow | DVC |
|---|---|---|---|
| Experiment tracking | ✅ file-backed JSON | ✅ SQLite/S3 | ❌ |
| Model registry | ✅ stage promotion + rollback | ✅ | ❌ |
| A/B testing | ✅ Welch t-test, p-value | ❌ | ❌ |
| Drift detection | ✅ PSI + KS | ❌ | ❌ |
| Retraining triggers | ✅ drift + metric threshold | ❌ | ✅ |
| Zero external MLOps deps | ✅ stdlib + numpy | ❌ | ❌ |
| Runs fully offline | ✅ | ❌ (server) | ✅ |

---

## Running the Tests

```bash
pytest -v --tb=short
```

The test suite covers:
- **Experiment Tracker** — metric logging, step history, persistence, best-run selection, multi-experiment isolation
- **Model Registry** — version bumping, stage transitions (all paths), production guard, rollback
- **A/B Test** — traffic split validation, Welch t-test math, significance, winner declaration
- **Drift Detection** — PSI calculation, KS statistic, threshold alerting, multi-feature monitor
- **Pipeline** — trigger conditions, full train→register→promote flow, edge cases (exceptions, no incumbent)

All tests are CPU-only with no network calls.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for four Mermaid diagrams:

1. Full MLOps lifecycle (train → track → register → serve → monitor → retrain)
2. Model promotion state machine (Staging → Production → Archived + rollback)
3. A/B testing decision flow
4. Drift detection pipeline

---

## Configuration

Edit `configs/platform_config.yaml` to tune thresholds, paths, and feature flags.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `drift_detection.psi_threshold` | `0.25` | PSI above which retraining is triggered |
| `drift_detection.ks_threshold` | `0.05` | KS p-value below which drift is flagged |
| `ab_testing.significance_alpha` | `0.05` | Statistical significance threshold |
| `ab_testing.min_samples_per_arm` | `100` | Minimum observations before declaring a winner |
| `pipeline.promote_threshold_pct` | `0.02` | Minimum relative improvement (2%) for promotion |
| `pipeline.retrain_metric_threshold` | `0.80` | Accuracy below this triggers retraining |

---

## License

MIT — see [LICENSE](LICENSE).
