"""Tests for drift detection (PSI, KS test, DriftMonitor)."""

import math
import pytest
from src.drift import (
    population_stability_index,
    ks_drift_score,
    psi_severity,
    DriftMonitor,
)


# ------------------------------------------------------------------
# PSI
# ------------------------------------------------------------------

def test_psi_identical_distributions():
    data = list(range(100))
    psi = population_stability_index(data, data)
    assert psi == pytest.approx(0.0, abs=0.01)


def test_psi_no_drift_label():
    data = list(range(100))
    psi = population_stability_index(data, data)
    assert psi_severity(psi) == "no_drift"


def test_psi_significant_drift():
    expected = list(range(0, 100))
    actual = list(range(200, 300))  # completely disjoint
    psi = population_stability_index(expected, actual)
    assert psi > 0.25


def test_psi_significant_label():
    assert psi_severity(0.30) == "significant_drift"


def test_psi_moderate_label():
    assert psi_severity(0.15) == "moderate_drift"


def test_psi_no_drift_label_explicit():
    assert psi_severity(0.05) == "no_drift"


def test_psi_small_shift():
    import random
    rng = random.Random(0)
    expected = [rng.gauss(0, 1) for _ in range(500)]
    actual = [rng.gauss(0.1, 1) for _ in range(500)]  # tiny shift
    psi = population_stability_index(expected, actual)
    assert psi < 0.25  # small shift should not trigger significant drift


def test_psi_large_shift():
    import random
    rng = random.Random(1)
    expected = [rng.gauss(0, 1) for _ in range(500)]
    actual = [rng.gauss(5, 1) for _ in range(500)]  # large shift
    psi = population_stability_index(expected, actual)
    assert psi > 0.25


def test_psi_empty_raises():
    with pytest.raises(ValueError):
        population_stability_index([], [1, 2, 3])


def test_psi_bins_zero_raises():
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [1, 2], bins=0)


def test_psi_custom_bins():
    data = list(range(200))
    psi = population_stability_index(data, data, bins=20)
    assert psi == pytest.approx(0.0, abs=0.05)


def test_psi_degenerate_single_value():
    # All same value — PSI should be 0 (no variance to measure)
    psi = population_stability_index([5.0] * 50, [5.0] * 50)
    assert psi == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------
# KS test
# ------------------------------------------------------------------

def test_ks_identical_distributions():
    data = [float(i) for i in range(50)]
    ks_stat, p_value = ks_drift_score(data, data)
    assert ks_stat == pytest.approx(0.0, abs=0.01)
    assert p_value > 0.05


def test_ks_different_distributions():
    ref = [float(i) for i in range(50)]
    cur = [float(i + 100) for i in range(50)]  # completely shifted
    ks_stat, p_value = ks_drift_score(ref, cur)
    assert ks_stat > 0.5
    assert p_value < 0.05


def test_ks_stat_range():
    ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    cur = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    ks_stat, p_value = ks_drift_score(ref, cur)
    assert 0.0 <= ks_stat <= 1.0
    assert 0.0 <= p_value <= 1.0


def test_ks_empty_raises():
    with pytest.raises(ValueError):
        ks_drift_score([], [1.0, 2.0])


# ------------------------------------------------------------------
# DriftMonitor
# ------------------------------------------------------------------

@pytest.fixture
def monitor():
    reference = {
        "feature_a": [float(i) for i in range(200)],
        "feature_b": [float(i) * 0.5 for i in range(200)],
    }
    return DriftMonitor(reference, psi_threshold=0.25, ks_threshold=0.05)


def test_drift_monitor_no_drift(monitor):
    current = {
        "feature_a": [float(i) for i in range(200)],
        "feature_b": [float(i) * 0.5 for i in range(200)],
    }
    report = monitor.check(current)
    assert report["overall_drifted"] is False


def test_drift_monitor_with_drift(monitor):
    current = {
        "feature_a": [float(i + 1000) for i in range(200)],  # huge shift
        "feature_b": [float(i) * 0.5 for i in range(200)],
    }
    report = monitor.check(current)
    assert report["overall_drifted"] is True
    assert report["feature_a"]["drifted"] is True


def test_drift_monitor_report_keys(monitor):
    current = {
        "feature_a": [float(i) for i in range(200)],
        "feature_b": [float(i) * 0.5 for i in range(200)],
    }
    report = monitor.check(current)
    for feature in ("feature_a", "feature_b"):
        assert "psi" in report[feature]
        assert "ks_stat" in report[feature]
        assert "drifted" in report[feature]
        assert "severity" in report[feature]


def test_drift_monitor_missing_feature_raises(monitor):
    with pytest.raises(KeyError, match="feature_a"):
        monitor.check({"feature_b": [1.0, 2.0, 3.0]})


def test_drift_monitor_empty_reference_raises():
    with pytest.raises(ValueError):
        DriftMonitor({})


def test_drift_monitor_psi_values_present(monitor):
    current = {
        "feature_a": [float(i) for i in range(200)],
        "feature_b": [float(i) * 0.5 for i in range(200)],
    }
    report = monitor.check(current)
    for feature in ("feature_a", "feature_b"):
        assert isinstance(report[feature]["psi"], float)
        assert report[feature]["psi"] >= 0
