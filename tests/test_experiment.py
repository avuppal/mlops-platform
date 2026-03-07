"""Tests for ExperimentTracker."""

import pytest
import tempfile
import os
from pathlib import Path
from src.experiment import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(tmp_path / "experiments")


# ------------------------------------------------------------------
# Run lifecycle
# ------------------------------------------------------------------

def test_start_run_returns_string_id(tracker):
    run_id = tracker.start_run("exp1")
    assert isinstance(run_id, str) and len(run_id) > 0


def test_start_run_with_params(tracker):
    run_id = tracker.start_run("exp1", params={"lr": 0.01, "epochs": 10})
    run = tracker.get_run(run_id)
    assert run["params"]["lr"] == 0.01
    assert run["params"]["epochs"] == 10


def test_start_run_status_is_running(tracker):
    run_id = tracker.start_run("exp1")
    run = tracker.get_run(run_id)
    assert run["status"] == "RUNNING"


def test_end_run_status_finished(tracker):
    run_id = tracker.start_run("exp1")
    tracker.end_run(run_id)
    run = tracker.get_run(run_id)
    assert run["status"] == "FINISHED"


def test_end_run_custom_status(tracker):
    run_id = tracker.start_run("exp1")
    tracker.end_run(run_id, status="FAILED")
    run = tracker.get_run(run_id)
    assert run["status"] == "FAILED"


def test_end_run_sets_end_time(tracker):
    run_id = tracker.start_run("exp1")
    tracker.end_run(run_id)
    run = tracker.get_run(run_id)
    assert run["end_time"] is not None
    assert run["end_time"] >= run["start_time"]


# ------------------------------------------------------------------
# Metric logging
# ------------------------------------------------------------------

def test_log_metric_single(tracker):
    run_id = tracker.start_run("exp1")
    tracker.log_metric(run_id, "accuracy", 0.92)
    run = tracker.get_run(run_id)
    assert run["metrics"]["accuracy"][0]["value"] == 0.92


def test_log_metric_step_recorded(tracker):
    run_id = tracker.start_run("exp1")
    tracker.log_metric(run_id, "loss", 0.5, step=1)
    tracker.log_metric(run_id, "loss", 0.3, step=2)
    run = tracker.get_run(run_id)
    steps = [e["step"] for e in run["metrics"]["loss"]]
    assert steps == [1, 2]


def test_log_metric_history_accumulates(tracker):
    run_id = tracker.start_run("exp1")
    for i in range(5):
        tracker.log_metric(run_id, "loss", 1.0 - i * 0.1, step=i)
    run = tracker.get_run(run_id)
    assert len(run["metrics"]["loss"]) == 5


def test_log_params_merges(tracker):
    run_id = tracker.start_run("exp1", params={"lr": 0.01})
    tracker.log_params(run_id, {"batch_size": 32, "optimizer": "adam"})
    run = tracker.get_run(run_id)
    assert run["params"]["lr"] == 0.01
    assert run["params"]["batch_size"] == 32
    assert run["params"]["optimizer"] == "adam"


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_persistence_across_instances(tmp_path):
    path = tmp_path / "store"
    t1 = ExperimentTracker(path)
    run_id = t1.start_run("myexp", params={"x": 42})
    t1.log_metric(run_id, "f1", 0.88)

    t2 = ExperimentTracker(path)
    run = t2.get_run(run_id)
    assert run["params"]["x"] == 42
    assert run["metrics"]["f1"][0]["value"] == 0.88


# ------------------------------------------------------------------
# Best run
# ------------------------------------------------------------------

def test_get_best_run_max(tracker):
    exp = "comparison"
    ids = []
    for acc in [0.80, 0.90, 0.85]:
        rid = tracker.start_run(exp)
        tracker.log_metric(rid, "accuracy", acc)
        tracker.end_run(rid)
        ids.append(rid)

    best = tracker.get_best_run(exp, "accuracy", mode="max")
    assert best["metrics"]["accuracy"][-1]["value"] == 0.90


def test_get_best_run_min(tracker):
    exp = "loss_exp"
    for loss in [0.5, 0.2, 0.35]:
        rid = tracker.start_run(exp)
        tracker.log_metric(rid, "loss", loss)
        tracker.end_run(rid)

    best = tracker.get_best_run(exp, "loss", mode="min")
    assert best["metrics"]["loss"][-1]["value"] == 0.2


def test_get_best_run_raises_on_unknown_metric(tracker):
    rid = tracker.start_run("e")
    tracker.log_metric(rid, "accuracy", 0.9)
    tracker.end_run(rid)
    with pytest.raises(ValueError, match="No runs"):
        tracker.get_best_run("e", "nonexistent_metric")


def test_get_best_run_invalid_mode(tracker):
    rid = tracker.start_run("e")
    tracker.log_metric(rid, "accuracy", 0.9)
    with pytest.raises(ValueError, match="mode must be"):
        tracker.get_best_run("e", "accuracy", mode="median")


# ------------------------------------------------------------------
# Compare runs
# ------------------------------------------------------------------

def test_compare_runs_structure(tracker):
    exp = "compare"
    run_ids = []
    for acc, loss in [(0.9, 0.1), (0.85, 0.15)]:
        rid = tracker.start_run(exp, params={"x": acc})
        tracker.log_metric(rid, "accuracy", acc)
        tracker.log_metric(rid, "loss", loss)
        tracker.end_run(rid)
        run_ids.append(rid)

    comparison = tracker.compare_runs(run_ids)
    assert set(comparison.keys()) == set(run_ids)
    for rid in run_ids:
        assert "accuracy" in comparison[rid]
        assert "loss" in comparison[rid]
        assert "params" in comparison[rid]


def test_compare_runs_values(tracker):
    exp = "cmp2"
    rid = tracker.start_run(exp)
    tracker.log_metric(rid, "accuracy", 0.77)
    tracker.end_run(rid)

    comparison = tracker.compare_runs([rid])
    assert comparison[rid]["accuracy"] == pytest.approx(0.77)


def test_list_runs(tracker):
    for i in range(3):
        rid = tracker.start_run("listexp")
        tracker.end_run(rid)
    runs = tracker.list_runs("listexp")
    assert len(runs) == 3


def test_multiple_experiments_isolated(tracker):
    r1 = tracker.start_run("expA")
    r2 = tracker.start_run("expB")
    tracker.log_metric(r1, "score", 1.0)
    tracker.log_metric(r2, "score", 2.0)

    runs_a = tracker.list_runs("expA")
    runs_b = tracker.list_runs("expB")
    assert len(runs_a) == 1
    assert len(runs_b) == 1


def test_unknown_run_id_raises(tracker):
    with pytest.raises(KeyError):
        tracker.get_run("nonexistent_run_id_xyz")
