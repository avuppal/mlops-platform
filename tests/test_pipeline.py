"""Tests for RetrainingPipeline."""

import pytest
from unittest.mock import MagicMock, patch
from src.experiment import ExperimentTracker
from src.registry import ModelRegistry
from src.drift import DriftMonitor
from src.pipeline import RetrainingPipeline


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(tmp_path / "experiments")


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


@pytest.fixture
def drift_monitor():
    reference = {"feat": [float(i) for i in range(100)]}
    return DriftMonitor(reference, psi_threshold=0.25, ks_threshold=0.05)


def make_pipeline(tracker, registry, drift_monitor, train_fn=None, eval_fn=None, **kwargs):
    train_fn = train_fn or (lambda data: {"weights": [1, 2, 3]})
    eval_fn = eval_fn or (lambda model, data: {"accuracy": 0.92})
    return RetrainingPipeline(
        tracker=tracker,
        registry=registry,
        drift_monitor=drift_monitor,
        train_fn=train_fn,
        eval_fn=eval_fn,
        **kwargs,
    )


# ------------------------------------------------------------------
# should_retrain
# ------------------------------------------------------------------

def test_should_retrain_on_drift(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    drift_report = {"overall_drifted": True}
    assert pipeline.should_retrain({"accuracy": 0.95}, drift_report) is True


def test_should_not_retrain_no_drift_good_metric(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    drift_report = {"overall_drifted": False}
    assert pipeline.should_retrain({"accuracy": 0.95}, drift_report) is False


def test_should_retrain_on_metric_degradation(tracker, registry, drift_monitor):
    pipeline = make_pipeline(
        tracker, registry, drift_monitor,
        metric="accuracy",
        retrain_metric_threshold=0.85,
    )
    drift_report = {"overall_drifted": False}
    # 0.80 < threshold 0.85 → retrain
    assert pipeline.should_retrain({"accuracy": 0.80}, drift_report) is True


def test_should_not_retrain_metric_above_threshold(tracker, registry, drift_monitor):
    pipeline = make_pipeline(
        tracker, registry, drift_monitor,
        metric="accuracy",
        retrain_metric_threshold=0.85,
    )
    drift_report = {"overall_drifted": False}
    assert pipeline.should_retrain({"accuracy": 0.90}, drift_report) is False


def test_should_retrain_metric_mode_min(tracker, registry, drift_monitor):
    pipeline = make_pipeline(
        tracker, registry, drift_monitor,
        metric="loss",
        metric_mode="min",
        retrain_metric_threshold=0.20,
    )
    drift_report = {"overall_drifted": False}
    # loss 0.30 > threshold 0.20 (too high) → retrain
    assert pipeline.should_retrain({"loss": 0.30}, drift_report) is True


def test_should_retrain_missing_metric(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    drift_report = {"overall_drifted": False}
    # metric not in current_metrics → no retrain trigger from metric
    assert pipeline.should_retrain({}, drift_report) is False


# ------------------------------------------------------------------
# Full pipeline run
# ------------------------------------------------------------------

def test_run_returns_correct_keys(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    result = pipeline.run("test_exp", train_data={}, eval_data={})
    for key in ("run_id", "version", "metrics", "promoted"):
        assert key in result


def test_run_registers_model_in_registry(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    result = pipeline.run("fraud_model", train_data={}, eval_data={})
    versions = registry.list_versions("fraud_model")
    assert len(versions) == 1
    assert versions[0]["version"] == result["version"]


def test_run_promotes_first_model_to_production(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    result = pipeline.run("my_model", train_data={}, eval_data={})
    # First model should always be promoted (no incumbent)
    assert result["promoted"] is True
    prod = registry.get_production_model("my_model")
    assert prod is not None


def test_run_logs_experiment(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    result = pipeline.run("tracked_exp", train_data={}, eval_data={})
    run = tracker.get_run(result["run_id"])
    assert run["status"] == "FINISHED"
    assert "accuracy" in run["metrics"]


def test_run_failed_on_train_exception(tracker, registry, drift_monitor):
    def bad_train(data):
        raise RuntimeError("GPU exploded")

    pipeline = make_pipeline(tracker, registry, drift_monitor, train_fn=bad_train)
    with pytest.raises(RuntimeError, match="Pipeline run failed"):
        pipeline.run("failing_exp", {}, {})


def test_run_history_tracked(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    pipeline.run("exp1", {}, {})
    pipeline.run("exp2", {}, {})
    assert len(pipeline.run_history) == 2


# ------------------------------------------------------------------
# promote_if_better
# ------------------------------------------------------------------

def test_promote_if_better_first_version(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    registry.register_model("m", "1.0.0", "path", metrics={"accuracy": 0.90})
    promoted = pipeline.promote_if_better("m", "1.0.0", "accuracy")
    assert promoted is True


def test_promote_if_better_significant_improvement(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    registry.register_model("m", "1.0.0", "path", metrics={"accuracy": 0.85})
    registry.register_model("m", "2.0.0", "path", metrics={"accuracy": 0.92})
    registry.promote("m", "1.0.0", "Production")

    promoted = pipeline.promote_if_better("m", "2.0.0", "accuracy", threshold_pct=0.02)
    assert promoted is True


def test_promote_if_better_insufficient_improvement(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    registry.register_model("m", "1.0.0", "path", metrics={"accuracy": 0.900})
    registry.register_model("m", "2.0.0", "path", metrics={"accuracy": 0.901})  # <2% gain
    registry.promote("m", "1.0.0", "Production")

    promoted = pipeline.promote_if_better("m", "2.0.0", "accuracy", threshold_pct=0.02)
    assert promoted is False


def test_promote_if_better_no_metric_returns_false(tracker, registry, drift_monitor):
    pipeline = make_pipeline(tracker, registry, drift_monitor)
    registry.register_model("m", "1.0.0", "path", metrics={})
    promoted = pipeline.promote_if_better("m", "1.0.0", "accuracy")
    assert promoted is False
